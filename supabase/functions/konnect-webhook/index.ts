import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// Doit rester identique à services/payment_provider.py::PLAN_PRICES_TND
// (vérifié par tests/test_payment_provider.py).
const PLAN_PRICES_TND: Record<string, Record<string, number>> = {
  pro: { monthly: 29, yearly: 299 },
  business: { monthly: 79, yearly: 799 },
  business_plus: { monthly: 149, yearly: 1499 },
};

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { "content-type": "application/json" },
});

class DbError extends Error {}

function check<T extends { error: unknown }>(res: T, what: string): T {
  if (res.error) {
    console.error(`[konnect-webhook] ${what}:`, res.error);
    throw new DbError(what);
  }
  return res;
}

function parseOrderId(orderId: string) {
  // autodeal:<uuid>:<plan>:<cycle>
  const parts = String(orderId || "").split(":");
  if (parts.length !== 4 || parts[0] !== "autodeal") return null;
  const [, userId, plan, cycle] = parts;
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(userId)) return null;
  if (!(plan in PLAN_PRICES_TND)) return null;
  if (!["monthly", "yearly"].includes(cycle)) return null;
  return { userId, plan, cycle };
}

// Konnect expose le wallet destinataire sous plusieurs formes selon la version.
function receiverWalletIds(p: Record<string, any>): string[] {
  return [
    p.receiverWalletId,
    p.receiverWallet?._id,
    p.receiverWallet?.id,
    p.receiverWallet,
    p.wallet?._id,
    p.wallet?.id,
  ].filter((v) => typeof v === "string" && v.length > 0);
}

// Ajoute n mois sans débordement (31 janv. + 1 mois = 28/29 févr., pas 3 mars).
function addMonthsClamped(date: Date, months: number): Date {
  const d = new Date(date);
  const day = d.getUTCDate();
  d.setUTCDate(1);
  d.setUTCMonth(d.getUTCMonth() + months);
  const lastDay = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0)).getUTCDate();
  d.setUTCDate(Math.min(day, lastDay));
  return d;
}

Deno.serve(async (req) => {
  try {
    const url = new URL(req.url);
    let paymentRef = url.searchParams.get("payment_ref");
    if (!paymentRef && req.method !== "GET") {
      const body = await req.json().catch(() => ({}));
      paymentRef = body.payment_ref || body.paymentRef || body.id || null;
    }
    if (!paymentRef) return json({ error: "payment_ref missing" }, 400);

    // On ne fait JAMAIS confiance au corps de la requête : l'état du paiement
    // est relu côté Konnect avec notre clé API.
    const apiKey = Deno.env.get("KONNECT_API_KEY")!;
    const walletId = Deno.env.get("KONNECT_WALLET_ID")!;
    const apiBase = (Deno.env.get("KONNECT_API_BASE_URL") || "https://api.konnect.network/api/v2").replace(/\/$/, "");
    const detailResp = await fetch(`${apiBase}/payments/${encodeURIComponent(paymentRef)}`, {
      headers: { "x-api-key": apiKey },
    });
    if (!detailResp.ok) return json({ error: "payment verification failed" }, 502);
    const details = await detailResp.json();
    const p = details.payment || details;
    const orderId = p.orderId || p.order_id || p?.details?.orderId || p?.paymentDetails?.orderId || "";
    const parsed = parseOrderId(orderId);
    if (!parsed) return json({ error: "unknown order" }, 400);

    // 1) Le paiement doit avoir été encaissé sur NOTRE wallet.
    if (!walletId || !receiverWalletIds(p).includes(walletId)) {
      console.error("[konnect-webhook] wallet mismatch", paymentRef);
      return json({ error: "wallet mismatch" }, 400);
    }

    const rawStatus = String(p.status || "").toLowerCase();
    const paid = ["completed", "paid", "success", "successful"].includes(rawStatus);
    // expired / cancelled = checkout abandonné : pas d'échec facturé, pas d'e-mail.
    const failed = ["failed", "rejected"].includes(rawStatus);
    const abandoned = ["cancelled", "canceled", "expired"].includes(rawStatus);

    // 2) Montant et devise conformes au prix de la formule.
    const expectedMillimes = PLAN_PRICES_TND[parsed.plan][parsed.cycle] * 1000;
    const amountMillimes = Number(p.reachedAmount ?? p.amount ?? p.amountDue ?? 0);
    const token = String(p.token || "TND").toUpperCase();
    if (paid && (token !== "TND" || !(amountMillimes >= expectedMillimes))) {
      console.error("[konnect-webhook] amount mismatch", paymentRef, amountMillimes, expectedMillimes, token);
      return json({ error: "amount mismatch" }, 400);
    }

    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
      { auth: { persistSession: false } },
    );
    const now = new Date();
    const amountTnd = amountMillimes ? amountMillimes / 1000 : null;

    // 3) Idempotence : si ce paiement est déjà traité, ne rien prolonger.
    const { data: txRows } = check(
      await supabase.from("payment_transactions").select("status").eq("payment_ref", paymentRef).limit(1),
      "read transaction",
    );
    const previousStatus = txRows?.[0]?.status;
    if (previousStatus === "paid") return json({ ok: true, status: rawStatus, duplicate: true });

    const txStatus = paid ? "pending" : failed ? "failed" : abandoned ? "expired" : "pending";
    check(
      await supabase.from("payment_transactions").upsert({
        user_id: parsed.userId,
        provider: "konnect",
        payment_ref: paymentRef,
        order_id: orderId,
        plan: parsed.plan,
        billing_cycle: parsed.cycle,
        currency: "TND",
        amount_tnd: amountTnd,
        status: txStatus,
        raw_payload: details,
        updated_at: now.toISOString(),
      }, { onConflict: "payment_ref" }),
      "upsert transaction",
    );

    const { data: existingRows } = check(
      await supabase.from("subscriptions").select("*").eq("user_id", parsed.userId).limit(1),
      "read subscription",
    );
    const existing = existingRows?.[0] || {};

    if (paid) {
      // Passage pending -> paid conditionnel : seul le premier appel concurrent
      // obtient la ligne et prolonge l'abonnement.
      const { data: claimed } = check(
        await supabase.from("payment_transactions")
          .update({ status: "paid", paid_at: now.toISOString(), updated_at: now.toISOString() })
          .eq("payment_ref", paymentRef)
          .neq("status", "paid")
          .select("id"),
        "claim transaction",
      );
      if (!claimed?.length) return json({ ok: true, status: rawStatus, duplicate: true });

      // Renouvellement anticipé de la même formule : on prolonge à partir de la
      // fin de période en cours au lieu de perdre les jours restants.
      const currentEnd = existing.current_period_end ? new Date(existing.current_period_end) : null;
      const samePlanActive = existing.plan === parsed.plan && existing.status === "active" && currentEnd && currentEnd > now;
      const periodStart = samePlanActive ? currentEnd! : now;
      const periodEnd = addMonthsClamped(periodStart, parsed.cycle === "yearly" ? 12 : 1);

      check(
        await supabase.from("subscriptions").upsert({
          user_id: parsed.userId,
          plan: parsed.plan,
          status: "active",
          billing_cycle: parsed.cycle,
          currency: "TND",
          provider: "konnect",
          payment_provider: "konnect",
          last_payment_status: "paid",
          last_payment_at: now.toISOString(),
          current_period_start: samePlanActive ? existing.current_period_start : periodStart.toISOString(),
          current_period_end: periodEnd.toISOString(),
          next_payment_due_at: periodEnd.toISOString(),
          failed_payment_count: 0,
          updated_at: now.toISOString(),
        }, { onConflict: "user_id" }),
        "upsert subscription",
      );

      check(
        await supabase.from("subscription_notifications").insert({
          user_id: parsed.userId,
          kind: "payment_success",
          payload: { plan: parsed.plan, billing_cycle: parsed.cycle, period_end: periodEnd.toISOString() },
        }),
        "insert notification",
      );
    } else if (failed && previousStatus !== "failed") {
      const previousEnd = existing.current_period_end ? new Date(existing.current_period_end) : null;
      const alreadyExpired = !previousEnd || previousEnd <= now;
      check(
        await supabase.from("subscriptions").update({
          status: alreadyExpired ? "past_due" : existing.status,
          last_payment_status: "failed",
          failed_payment_count: Number(existing.failed_payment_count || 0) + 1,
          updated_at: now.toISOString(),
        }).eq("user_id", parsed.userId),
        "update subscription",
      );

      check(
        await supabase.from("subscription_notifications").insert({
          user_id: parsed.userId,
          kind: "payment_failed",
          payload: { plan: parsed.plan, billing_cycle: parsed.cycle, access_locked: alreadyExpired },
        }),
        "insert notification",
      );
    }

    return json({ ok: true, status: rawStatus });
  } catch (error) {
    // 5xx => Konnect retente. Détail uniquement dans les logs serveur.
    console.error("[konnect-webhook] unexpected error", error);
    return json({ error: error instanceof DbError ? "storage error" : "internal error" }, 500);
  }
});

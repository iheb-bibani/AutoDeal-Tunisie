import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { "content-type": "application/json" },
});

function parseOrderId(orderId: string) {
  // autodeal:<uuid>:<plan>:<cycle>
  const parts = String(orderId || "").split(":");
  if (parts.length !== 4 || parts[0] !== "autodeal") return null;
  const [, userId, plan, cycle] = parts;
  if (!["pro", "business", "business_plus"].includes(plan)) return null;
  if (!["monthly", "yearly"].includes(cycle)) return null;
  return { userId, plan, cycle };
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

    const apiKey = Deno.env.get("KONNECT_API_KEY")!;
    const apiBase = (Deno.env.get("KONNECT_API_BASE_URL") || "https://api.konnect.network/api/v2").replace(/\/$/, "");
    const detailResp = await fetch(`${apiBase}/payments/${encodeURIComponent(paymentRef)}`, {
      headers: { "x-api-key": apiKey },
    });
    if (!detailResp.ok) return json({ error: "payment verification failed" }, 502);
    const details = await detailResp.json();
    const p = details.payment || details;
    const orderId = p.orderId || p.order_id || p?.details?.orderId || p?.paymentDetails?.orderId || "";
    const parsed = parseOrderId(orderId);
    if (!parsed) return json({ error: "unknown orderId", orderId }, 400);

    const rawStatus = String(p.status || "").toLowerCase();
    const paid = ["completed", "paid", "success", "successful"].includes(rawStatus);
    const failed = ["failed", "rejected", "cancelled", "canceled", "expired"].includes(rawStatus);

    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
      { auth: { persistSession: false } },
    );

    const now = new Date();
    const amountMillimes = Number(p.amount || p.amountDue || 0);
    const amountTnd = amountMillimes ? amountMillimes / 1000 : null;

    await supabase.from("payment_transactions").upsert({
      user_id: parsed.userId,
      provider: "konnect",
      payment_ref: paymentRef,
      order_id: orderId,
      plan: parsed.plan,
      billing_cycle: parsed.cycle,
      currency: "TND",
      amount_tnd: amountTnd,
      status: paid ? "paid" : failed ? "failed" : "pending",
      raw_payload: details,
      paid_at: paid ? now.toISOString() : null,
      updated_at: now.toISOString(),
    }, { onConflict: "payment_ref" });

    const { data: existingRows } = await supabase.from("subscriptions").select("*").eq("user_id", parsed.userId).limit(1);
    const existing = existingRows?.[0] || {};

    if (paid) {
      const periodStart = now;
      const periodEnd = new Date(now);
      if (parsed.cycle === "yearly") periodEnd.setUTCFullYear(periodEnd.getUTCFullYear() + 1);
      else periodEnd.setUTCMonth(periodEnd.getUTCMonth() + 1);

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
        current_period_start: periodStart.toISOString(),
        current_period_end: periodEnd.toISOString(),
        next_payment_due_at: periodEnd.toISOString(),
        failed_payment_count: 0,
        updated_at: now.toISOString(),
      }, { onConflict: "user_id" });

      await supabase.from("subscription_notifications").insert({
        user_id: parsed.userId,
        kind: "payment_success",
        payload: { plan: parsed.plan, billing_cycle: parsed.cycle, period_end: periodEnd.toISOString() },
      });
    } else if (failed) {
      const previousEnd = existing.current_period_end ? new Date(existing.current_period_end) : null;
      const alreadyExpired = !previousEnd || previousEnd <= now;
      await supabase.from("subscriptions").update({
        status: alreadyExpired ? "past_due" : existing.status,
        last_payment_status: "failed",
        failed_payment_count: Number(existing.failed_payment_count || 0) + 1,
        updated_at: now.toISOString(),
      }).eq("user_id", parsed.userId);

      await supabase.from("subscription_notifications").insert({
        user_id: parsed.userId,
        kind: "payment_failed",
        payload: { plan: parsed.plan, billing_cycle: parsed.cycle, access_locked: alreadyExpired },
      });
    }

    return json({ ok: true, status: rawStatus });
  } catch (error) {
    return json({ error: String(error) }, 500);
  }
});

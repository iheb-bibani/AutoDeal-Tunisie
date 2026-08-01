"""Audit des 5 sources : schéma, complétude et survie dans le pipeline."""
from pathlib import Path
import sys
import json
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SCRAPERS, PROCESSED_FILES

ALIASES = {
    "boite": ["Boite", "Boite_Vitesse", "Transmission", "Gearbox"],
    "energie": ["Energie", "Carburant", "Fuel"],
    "kilometrage": ["Kilométrage", "Kilometrage", "Km"],
    "date_depot": ["Annonce-Deposee", "Date", "Date_Publication"],
}

def _read(path): return pd.read_csv(path, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
def _canon_source(x):
    x=str(x).lower()
    if "tayara" in x:return "tayara"
    if "automobile" in x:return "automobile"
    if "automax" in x:return "automax"
    if "autocentral" in x:return "autocentral"
    if "sayyarat" in x:return "sayyarat"
    return x

def main():
    recent=_read(PROCESSED_FILES["recent"])
    final=_read(PROCESSED_FILES["enriched"])
    rows=[]
    for name,path in SCRAPERS.items():
        if not Path(path).exists(): continue
        d=_read(path)
        boite_col=next((c for c in ["Boite","Boite_Vitesse"] if c in d),None)
        boite=d[boite_col] if boite_col else pd.Series(index=d.index,dtype=object)
        b=boite.astype("string").str.lower()
        man=b.str.contains("manuelle|manual|bvm",regex=True,na=False).sum()
        aut=b.str.contains("automatique|automatic|bva|auto|dsg|cvt",regex=True,na=False).sum()
        r=recent[recent.Source.map(_canon_source).eq(name)] if "Source" in recent else recent.iloc[0:0]
        f=final[final.Source.map(_canon_source).eq(name)] if "Source" in final else final.iloc[0:0]
        valid=int(man+aut)
        rows.append({"source":name,"raw":len(d),"recent":len(r),"final_ml":len(f),
                     "boite_colonne":boite_col,"boite_valide_pct":round(valid/max(len(d),1)*100,1),
                     "manuelle":int(man),"automatique":int(aut),
                     "date_depot_complete_pct":round(d.get("Annonce-Deposee",pd.Series(index=d.index)).notna().mean()*100,1)})
    out={"sources":rows}
    Path("data/processed/audit_sources.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n-> data/processed/audit_sources.json")
if __name__=="__main__":main()

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parents[1]
CLEANED_DIR=ROOT/"outputs"/"cleaned"; RESULTS_DIR=ROOT/"outputs"/"results"; FIGURES_DIR=ROOT/"outputs"/"figures"
RESULTS_DIR.mkdir(parents=True,exist_ok=True); FIGURES_DIR.mkdir(parents=True,exist_ok=True)
RANDOM_STATE=42

def prep(num,cat):
    return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                              ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)],remainder="drop")
def model(est,num,cat): return Pipeline([("pre",prep(num,cat)),("model",est)])
def met(y,p):
    y=np.asarray(y); p=np.asarray(p); nz=np.abs(y)>1e-8; den=np.abs(y)+np.abs(p); valid=den>1e-8
    return {"MAE":mean_absolute_error(y,p),"RMSE":np.sqrt(mean_squared_error(y,p)),
            "MAPE":np.mean(np.abs((y[nz]-p[nz])/y[nz]))*100 if nz.any() else np.nan,
            "sMAPE":np.mean(2*np.abs(p[valid]-y[valid])/den[valid])*100 if valid.any() else np.nan,
            "R2":r2_score(y,p)}
def rank(df):
    for c,a in [("MAE",1),("RMSE",1),("MAPE",1),("sMAPE",1),("R2",0)]:
        df[c+" Rank"]=df[c].rank(ascending=bool(a),method="average")
    df["Average Rank"]=df[[c+" Rank" for c in ["MAE","RMSE","MAPE","sMAPE","R2"]]].mean(axis=1)
    df["Overall Rank"]=df["Average Rank"].rank(method="min").astype(int)
    return df.sort_values(["Overall Rank","RMSE"]).reset_index(drop=True)

def main():
    print("="*70); print("D1 - SUPERSTORE SALES"); print("MULTIPLE HYBRID COMPARATIVE ANALYSIS"); print("="*70)
    path=CLEANED_DIR/"superstore_cleaned.csv"
    df=pd.read_csv(path)
    df=df.rename(columns={c:str(c).strip().lower().replace("-","_").replace(" ","_") for c in df.columns})
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=["profit"]).reset_index(drop=True)
    print(f"\nLoading dataset: {path.name}"); print(f"Dataset rows: {len(df):,}"); print(f"Dataset columns: {len(df.columns)}"); print("Target: profit")
    allnum=[c for c in ["sales","quantity","discount","shipping_days","order_year","order_month"] if c in df.columns]
    allcat=[c for c in ["category","subcategory","region","stateprovince","segment","ship_mode"] if c in df.columns]
    selnum=[c for c in ["sales","discount","quantity","order_month"] if c in df.columns]
    selcat=[c for c in ["subcategory","stateprovince","category","region"] if c in df.columns]
    rng=np.random.RandomState(42); idx=np.arange(len(df)); rng.shuffle(idx); n=len(df); nt=int(n*.64); nv=int(n*.16)
    tr=df.iloc[idx[:nt]]; va=df.iloc[idx[nt:nt+nv]]; te=df.iloc[idx[nt+nv:]]
    yt=tr.profit.values; yv=va.profit.values; ys=te.profit.values
    print("\nData split:"); print(f"Training   : {len(tr):,}"); print(f"Validation : {len(va):,}"); print(f"Testing    : {len(te):,}")
    print("\nMODEL 1: NAIVE BASELINE"); nvpred=np.full(len(va),yt.mean()); nspred=np.full(len(te),yt.mean())
    print("MODEL 2: RIDGE REGRESSION"); ridge=model(Ridge(alpha=10),selnum,selcat); ridge.fit(tr,yt); rv=ridge.predict(va); rs=ridge.predict(te)
    print("MODEL 3: RANDOM FOREST - ALL FEATURES"); rfa=model(RandomForestRegressor(n_estimators=250,max_features="sqrt",min_samples_leaf=1,random_state=42,n_jobs=-1),allnum,allcat); rfa.fit(tr,yt); rav=rfa.predict(va); ras=rfa.predict(te)
    print("MODEL 4: RANDOM FOREST - SELECTED FEATURES"); print("Selected features:"); [print("  -",c) for c in selnum+selcat]
    rf=model(RandomForestRegressor(n_estimators=250,max_features="sqrt",min_samples_leaf=1,random_state=42,n_jobs=-1),selnum,selcat); rf.fit(tr,yt); rfv=rf.predict(va); rfs=rf.predict(te)
    print("MODEL 5: GRADIENT BOOSTING"); gb=model(GradientBoostingRegressor(n_estimators=250,learning_rate=.05,max_depth=3,loss="huber",random_state=42),selnum,selcat); gb.fit(tr,yt); gbv=gb.predict(va); gbs=gb.predict(te)
    print("\nHYBRID 1: RF + GB"); h1v=.5*rfv+.5*gbv; h1s=.5*rfs+.5*gbs
    print("HYBRID 2: Ridge + RF"); h2v=.5*rv+.5*rfv; h2s=.5*rs+.5*rfs
    print("HYBRID 3: Ridge + GB"); h3v=.5*rv+.5*gbv; h3s=.5*rs+.5*gbs
    print("HYBRID 4: RF + GB + Ridge"); h4v=(rfv+gbv+rv)/3; h4s=(rfs+gbs+rs)/3
    print("HYBRID 5: VALIDATION-OPTIMIZED ENSEMBLE")
    rows=[]; best=(0,0,1); br=np.inf
    for wr in np.arange(0,1.0001,.05):
        for wg in np.arange(0,1.0001-wr,.05):
            wri=1-wr-wg; p=wr*rfv+wg*gbv+wri*rv; score=np.sqrt(mean_squared_error(yv,p)); rows.append({"RF_Weight":round(wr,2),"GB_Weight":round(wg,2),"Ridge_Weight":round(wri,2),"Validation_RMSE":score})
            if score<br: br=score; best=(float(wr),float(wg),float(wri))
    pd.DataFrame(rows).sort_values("Validation_RMSE").to_csv(RESULTS_DIR/"d1_hybrid_weight_search.csv",index=False)
    wr,wg,wri=best; print(f"Best validation weights: RF={wr:.2f}, GB={wg:.2f}, Ridge={wri:.2f}")
    h5v=wr*rfv+wg*gbv+wri*rv; h5s=wr*rfs+wg*gbs+wri*rs
    print("HYBRID 6: GB + RIDGE RESIDUAL CORRECTION"); resid=yt-gb.predict(tr); rr=model(Ridge(alpha=10),selnum,selcat); rr.fit(tr,resid); rrv=rr.predict(va); rrs=rr.predict(te)
    ba=0; brr=np.inf
    for a in np.arange(0,1.01,.1):
        score=np.sqrt(mean_squared_error(yv,gbv+a*rrv))
        if score<brr: brr=score; ba=float(a)
    print(f"Best residual correction strength: {ba:.2f}"); h6v=gbv+ba*rrv; h6s=gbs+ba*rrs
    models={"Naive Model":(nvpred,nspred,"Baseline"),"Ridge Regression":(rv,rs,"Baseline"),"Random Forest":(rav,ras,"Baseline"),"Random Forest - Selected Features":(rfv,rfs,"Baseline"),"Gradient Boosting":(gbv,gbs,"Baseline"),"Hybrid 1 - RF + GB":(h1v,h1s,"Hybrid Candidate"),"Hybrid 2 - Ridge + RF":(h2v,h2s,"Hybrid Candidate"),"Hybrid 3 - Ridge + GB":(h3v,h3s,"Hybrid Candidate"),"Hybrid 4 - RF + GB + Ridge":(h4v,h4s,"Hybrid Candidate"),"Hybrid 5 - Validation Optimized Ensemble":(h5v,h5s,"Hybrid Candidate"),"Hybrid 6 - GB + Ridge Residual Correction":(h6v,h6s,"Hybrid Candidate")}
    vr=[]; sr=[]; pr=[]
    for name,(pv,ps,fam) in models.items():
        vr.append({"Model":name,"Family":fam,**met(yv,pv)}); sr.append({"Model":name,"Family":fam,**met(ys,ps)})
        pr += [{"Model":name,"Actual":ys[i],"Prediction":ps[i],"Residual":ys[i]-ps[i]} for i in range(len(ys))]
    vr=rank(pd.DataFrame(vr)); sr=rank(pd.DataFrame(sr))
    vr.to_csv(RESULTS_DIR/"d1_hybrid_validation_results.csv",index=False); sr.to_csv(RESULTS_DIR/"d1_comparative_model_results.csv",index=False); pd.DataFrame(pr).to_csv(RESULTS_DIR/"d1_comparative_model_predictions.csv",index=False)
    print("\n"+"="*70); print("FINAL D1 TEST MODEL COMPARISON"); print("="*70)
    cols=["Model","Family","MAE","RMSE","MAPE","sMAPE","R2","Average Rank","Overall Rank"]; print(sr[cols].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    best=sr.iloc[0]; hy=sr[sr.Family=="Hybrid Candidate"].iloc[0]
    print("\n"+"="*70); print("BEST OVERALL MODEL"); print("="*70); print(f"Model: {best.Model}"); print(f"MAE: {best.MAE:.6f}"); print(f"RMSE: {best.RMSE:.6f}"); print(f"MAPE: {best.MAPE:.6f}"); print(f"sMAPE: {best.sMAPE:.6f}"); print(f"R2: {best.R2:.6f}")
    print("\n"+"="*70); print("BEST HYBRID CANDIDATE"); print("="*70); print(f"Model: {hy.Model}"); print(f"MAE: {hy.MAE:.6f}"); print(f"RMSE: {hy.RMSE:.6f}"); print(f"MAPE: {hy.MAPE:.6f}"); print(f"sMAPE: {hy.sMAPE:.6f}"); print(f"R2: {hy.R2:.6f}")
    x=np.arange(len(sr)); labels=sr.Model.str.replace(" - ","\n",regex=False); fig,ax=plt.subplots(figsize=(17,8)); ax.bar(x,sr.RMSE); ax.set_title("D1 Superstore - RMSE Model Comparison"); ax.set_ylabel("RMSE"); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=60,ha="right"); fig.tight_layout(); fig.savefig(FIGURES_DIR/"31_d1_comparative_analysis.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    hp=sr[sr.Family=="Hybrid Candidate"]; x=np.arange(len(hp)); fig,ax=plt.subplots(figsize=(14,7)); ax.bar(x,hp.RMSE); ax.set_title("D1 Hybrid Candidate RMSE Comparison"); ax.set_ylabel("RMSE"); ax.set_xticks(x); ax.set_xticklabels(hp.Model.str.replace("Hybrid ","H",regex=False),rotation=45,ha="right"); fig.tight_layout(); fig.savefig(FIGURES_DIR/"32_d1_hybrid_candidate_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,7)); x=np.arange(len(sr)); ax.bar(x,sr.R2); ax.set_title("D1 R2 Model Comparison"); ax.set_ylabel("R2"); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=60,ha="right"); fig.tight_layout(); fig.savefig(FIGURES_DIR/"33_d1_model_r2_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    print("\nSaved D1 results to outputs/results/"); print("Saved D1 figures 31, 32 and 33 to outputs/figures/"); print("\nD1 comparative analysis completed successfully.")

if __name__=="__main__": main()

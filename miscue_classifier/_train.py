"""Train the deployable miscue classifier on features that segment_features.py emits.
Critically: costed_neglog is recomputed with matrices/confusion_hindi_expt.npz -- the SAME
matrix segment_features.py uses -- so the served feature is on the trained scale."""
import pandas as pd, numpy as np, json, os, joblib, datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
R='/home/daplab/30006664/github_segment_features/'; OUT=R+'miscue_classifier/'
W='/home/daplab/30006664/wav2vec_deb/'
L=open(W+'Confusion_matrix/classification_edit_and_features_final.py',encoding='utf-8').read().split('\n')
cut=[i for i,l in enumerate(L) if l.startswith('def main(')][0]
g={}; exec(compile('\n'.join(L[:cut]),'x','exec'),g)
P=g['parse_phones']; ED=g['editdistance']
z=np.load(R+'matrices/confusion_hindi_expt.npz',allow_pickle=True)
CM=z['cost_neglog']; PH=[str(p) for p in z['phones']]; LAB={p:i for i,p in enumerate(PH)}; EPS=int(z['eps_idx'])
print(f"cost matrix {CM.shape} from confusion_hindi_expt.npz (n_pairs={int(z['n_pairs'])}, laplace={float(z['laplace'])})",flush=True)
MF=W+'Master_file_KV_2023/master_file/features_files/Master_file_with_36_SF_GOP_features_verified.csv'
SYS=['MT_decoded','Cano_decoded']
df=pd.read_csv(MF,usecols=['fold','GT_binary','canonical_phone_seq']+
   [f'{s}_{x}' for s in SYS for x in ('label','wav2vec_phone_seq_clean','entropy_gibbs_exp_min')],encoding='utf-8-sig')
df=df[df.GT_binary.isin([0,1])].reset_index(drop=True)
y=df.GT_binary.astype(int).values; folds=sorted(df.fold.unique()); fa=df.fold.values
ref=[P(v) for v in df.canonical_phone_seq.values]
HP=dict(learning_rate=0.06,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1.0,
        class_weight='balanced',random_state=42,early_stopping=True,max_iter=500)
def mk(): return HistGradientBoostingClassifier(**HP)
def met(yy,sc):
    fpr,tpr,thr=roc_curve(yy,sc); r=float(np.interp(0.05,fpr,tpr))
    t5=float(np.interp(0.05,fpr,thr[:len(fpr)]))
    TP=r*yy.sum(); FP=0.05*(yy==0).sum(); p=TP/(TP+FP)
    return dict(AUC=float(roc_auc_score(yy,sc)),AP=float(average_precision_score(yy,sc)),
                MR_at_5FPR=100*(1-r),F1=2*p*r/(p+r),Precision=p,Recall=r,threshold=t5)
man={}
for s in SYS:
    hyp=[P(v) for v in df[f'{s}_wav2vec_phone_seq_clean'].values]
    st={'oov_tokens':0,'total_tokens':0}; cn=np.full(len(df),np.nan)
    for i in range(len(df)):
        if hyp[i] and ref[i]:
            cn[i]=ED(hyp[i],ref[i],apply_sub_cost=True,cost_matrix=CM,labels=LAB,eps_idx=EPS,oov_stats=st)[0]
    conf=pd.to_numeric(df[f'{s}_entropy_gibbs_exp_min'],errors='coerce').values
    cs=np.isin(df[f'{s}_label'].values,['c','s'])
    for tag,names,X in (('ed1',['costed_neglog'],cn.reshape(-1,1)),
                        ('ed_conf2',['costed_neglog','entropy_gibbs_exp_min'],np.column_stack([cn,conf]))):
        o=np.full(len(df),np.nan)
        for h in folds:
            tr=(fa!=h)&cs; te=(fa==h)&cs
            m=mk(); m.fit(X[tr],y[tr]); i=np.where(te)[0]; o[i]=m.predict_proba(X[i])[:,1]
        mm=~np.isnan(o); M=met(y[mm],o[mm])
        final=mk(); final.fit(X[cs],y[cs])
        art=dict(model=final,feature_names=names,threshold_5pct_fpr=M['threshold'],hyperparameters=HP,
                 cost_matrix_source='matrices/confusion_hindi_expt.npz (key cost_neglog)',
                 target='1 = miscue (S), 0 = correct (C)',population='ASR label in {c,s}',
                 trained_on=f'KV-1908 master file, {s} timestamps, {int(cs.sum())} c/s rows',
                 cv_metrics=M,sklearn_note='pickled under scikit-learn 1.7.2',
                 built=datetime.datetime.now().isoformat(timespec='seconds'))
        f=f'{OUT}models/miscue_clf_{s}_{tag}.joblib'; joblib.dump(art,f,compress=3)
        man[f'{s}_{tag}']=dict(file=f'models/miscue_clf_{s}_{tag}.joblib',features=names,
            **{k:(round(v,4) if isinstance(v,float) else v) for k,v in M.items()},n_train=int(cs.sum()))
        print(f'  {s:13} {tag:9} AUC {M["AUC"]:.4f}  AP {M["AP"]:.4f}  MR@5%FPR {M["MR_at_5FPR"]:.2f}  thr {M["threshold"]:.4f}',flush=True)
json.dump(man,open(OUT+'manifest.json','w'),indent=2)
print('\n'+pd.DataFrame(man).T.to_string())

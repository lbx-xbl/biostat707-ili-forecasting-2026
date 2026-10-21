"""Evaluation is the only consumer of held-out target labels."""
import numpy as np
import pandas as pd

MODELS=['Persistence','Ridge','HGB','LSTM']
def scores(d):
    err=d.y_pred.to_numpy()-d.y_true.to_numpy()
    macro=d.assign(abs_error=np.abs(err)).groupby('region').abs_error.mean().mean()
    return dict(n=len(d),macro_mae=float(macro),pooled_mae=float(np.mean(np.abs(err))),pooled_rmse=float(np.sqrt(np.mean(err**2))))

def improvement(base,value):
    return 100*(base-value)/base if base!=0 else float('nan')

def validate_predictions(d,expected):
    if set(d.model)!=set(MODELS): raise ValueError('Unexpected model set')
    if d.duplicated(['model','sample_id']).any(): raise ValueError('Duplicate prediction')
    if not np.isfinite(d.y_pred).all(): raise ValueError('Nonfinite predictions')
    for model,g in d.groupby('model'):
        if set(g.sample_id)!=set(expected) or len(g)!=len(expected): raise ValueError(f'Prediction keys missing/extra: {model}')

def metrics(d):
    rows=[]
    for split,subset in d.groupby('split',sort=False):
        groups=[('overall','all',subset)]
        groups += [('region',r,g) for r,g in subset.groupby('region',sort=False)]
        groups += [('season',r,g) for r,g in subset.groupby('target_season',sort=False)]
        for scope,group,g in groups:
            base=scores(g[g.model=='Persistence'])['macro_mae']
            for model,m in g.groupby('model',sort=False):
                score=scores(m)
                rows.append(dict(split=split,scope=scope,group=group,model=model,**score,
                                 improvement_pct=improvement(base,score['macro_mae']),
                                 negative_predictions=int((m.y_pred<0).sum()),units='percentage points'))
    return pd.DataFrame(rows)

def render_report(d,m,curve,selection,summary,audit,out):
    import base64
    import html
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    out=Path(out); figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    colors={'Observed':'#18212b','Persistence':'#999999','Ridge':'#2376bd','HGB':'#d17a15','LSTM':'#9a3ca6'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    test=d[d.split=='Test'].copy(); test['target_date']=pd.to_datetime(test.target_date)
    fig,axes=plt.subplots(5,2,figsize=(15,16),sharex=True,layout='constrained')
    for r,ax in zip([f'hhs{i}' for i in range(1,11)],axes.flat):
        g=test[test.region==r]; obs=g[g.model=='Persistence'].sort_values('target_date')
        ax.plot(obs.target_date,obs.y_true,color=colors['Observed'],lw=1.8,label='Observed')
        for model in MODELS:
            p=g[g.model==model].sort_values('target_date')
            ax.plot(p.target_date,p.y_pred,color=colors[model],lw=1,alpha=.85,label=model)
        ax.set_title(r.upper()); ax.set_ylabel('Weighted ILI (%)'); ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=5)
    fig.supxlabel('Target week ending date')
    fig.savefig(figdir/'test_timeseries.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    overall=m[(m.split=='Test') & (m.scope=='overall')].set_index('model').loc[MODELS]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    bars=ax.bar(MODELS,overall.macro_mae,color=[colors[n] for n in MODELS])
    ax.bar_label(bars,fmt='%.4f',padding=5); ax.set_ylim(0,overall.macro_mae.max()*1.2)
    ax.set_ylabel('Macro-MAE (percentage points)'); ax.set_title('Test error: equal weight for each HHS region')
    fig.savefig(figdir/'model_comparison.png',dpi=150); plt.close(fig)
    lo=min(0,float(test.y_pred.min()),float(test.y_true.min())); hi=max(float(test.y_pred.max()),float(test.y_true.max()))*1.04
    fig,axes=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True,layout='constrained')
    for model,ax in zip(MODELS,axes.flat):
        g=test[test.model==model]
        ax.scatter(g.y_true,g.y_pred,s=9,alpha=.35,color=colors[model]); ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),title=model,xlabel='Observed weighted ILI (%)',ylabel='Predicted weighted ILI (%)'); ax.set_aspect('equal')
    fig.suptitle('Test predicted versus observed (bias diagnostic)')
    fig.savefig(figdir/'predicted_vs_observed.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(curve.epoch,curve.train_mse_standardized,color=colors['LSTM']); axes[0].set(xlabel='Epoch',ylabel='Training MSE (standardized units)')
    axes[1].plot(curve.epoch,curve.validation_macro_mae,color=colors['LSTM']); axes[1].set(xlabel='Epoch',ylabel='Validation Macro-MAE (percentage points)')
    for ax in axes: ax.axvline(selection['lstm']['best_epoch'],color='black',ls='--',label='Restored best epoch'); ax.legend()
    fig.savefig(figdir/'lstm_training.png',dpi=150); plt.close(fig)
    def table(frame): return frame.to_html(index=False,float_format=lambda v:f'{v:.6f}',border=0)
    def image(name,caption):
        b64=base64.b64encode((figdir/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    best=overall.macro_mae.idxmin(); lstm=overall.loc['LSTM']; baseline=overall.loc['Persistence']
    comparison='lower' if lstm.macro_mae<baseline.macro_mae else 'higher'
    curve_text=f"Validation MAE was {curve.validation_macro_mae.iloc[0]:.4f} at epoch 1 and {curve.validation_macro_mae.iloc[-1]:.4f} at the final epoch. The restored epoch was {selection['lstm']['best_epoch']} of {len(curve)}. Training MSE changed from {curve.train_mse_standardized.iloc[0]:.4f} to {curve.train_mse_standardized.iloc[-1]:.4f}."
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Regional ILI Forecasting Demo</title>
<style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#18212b}}h1{{line-height:1.2}}h2{{margin-top:38px}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}th,td{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#eef3f7}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#555}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:15px}}.note{{background:#eef4f9;padding:16px}}</style>
<h1>Regional ILI Forecasting Demo: Persistence, Ridge, Gradient Boosting, and LSTM</h1>
<p class="note">Completed retrospective demo using revised historical ILINet data. CPU, fixed seed 2026. No live forecasting, probabilistic calibration, or significance claims.</p>
<h2>Research question and data</h2><p>Can a small pooled LSTM improve two-observation-week-ahead regional weighted ILI predictions over persistence, regularized linear regression, and gradient boosting?</p>
<p>CDC ILINet data provided by CMU Delphi: {audit['rows']:,} rows, ten HHS regions, 469 consecutive CDC/MMWR weeks per region (2010w40–2019w39). wili is weighted ILI outpatient visit percentage: 2.5 means 2.5%; error units are percentage points. This is neither laboratory-confirmed influenza nor population infection prevalence.</p>
<p>Downloaded {html.escape(audit['source_snapshot']['fetched_at_utc'])}, HTTP 200, API result=1. Snapshot SHA-256: <code>{audit['source_snapshot']['sha256']}</code>. Observed release dates range {audit['release_date_range']}; issue values range {audit['issue_range']}. These metadata are preserved but excluded from features. Public Domain US Government source.</p>
<h2>Time split and information restrictions</h2><p>Each sample uses 12 consecutive regional observations ending at origin and targets origin + 14 days. Both origin and target must belong to one split. Earlier-split history may be used; the last two origins before a boundary are excluded. Week 53 is retained. Test includes two continuous seasons; later Test inputs can use earlier observed Test values, without refitting.</p>
{table(summary)}
<p>The global scaler uses {selection['transform']['n_unique_observations']} unique Train region-week observations, mean {selection['transform']['mu']:.8f}, population SD {selection['transform']['sigma']:.8f}. It transforms histories and training targets only; one-hot and calendar features remain unchanged. Ridge/tree receive 24 features; LSTM receives a 12×1 sequence and the same 12 static features. Annual sine/cosine use target-date days since 2000-01-01 divided by 365.2425.</p>
<h2>Models and selection</h2><p>Persistence copies the last raw observation. All learned models are pooled across regions and fitted only on Train. No Train+Validation refit. Ridge uses validation region-equal MAE and larger alpha on exact ties. The fixed tree has no internal early stopping. The LSTM resets hidden state per window, uses an LSTM(1,16) and a linear head on hidden state plus static features, MSE/Adam, shuffled Train batches, gradient clipping and validation-only stopping. No Test labels enter training or selection.</p>
{table(pd.DataFrame(selection['ridge_candidates']))}
<p>Selected Ridge alpha: {selection['ridge_selected_alpha']}. LSTM: best epoch {selection['lstm']['best_epoch']}, total epochs {selection['lstm']['total_epochs']}. Early stopping triggered: {selection['lstm']['early_stop_triggered']}.</p>
<details><summary>Full frozen configuration and preprocessing</summary><pre>{html.escape(json.dumps(selection,indent=2))}</pre></details>
<h2>Validation and Test results</h2><p>Macro-MAE averages regional MAEs equally. Pooled MAE/RMSE weight samples equally. Improvement (%) = 100 × (Persistence Macro-MAE − model Macro-MAE) / Persistence Macro-MAE. A zero baseline yields NA. All four models share 500 Validation and 1,020 Test samples. Each model is fitted using 3,000 Train windows; Train is not a held-out evaluation set. Predictions are never clipped.</p>
{table(m[m.scope=='overall'])}
<p>The lowest Test Macro-MAE was obtained by {best}. LSTM Test Macro-MAE was {comparison} than Persistence ({lstm.macro_mae:.4f} versus {baseline.macro_mae:.4f} percentage points; improvement {lstm.improvement_pct:.2f}%). This is a descriptive comparison in one fixed historical split. {curve_text} The curve documents optimization and validation selection, but does not by itself identify why a model generalizes differently. The small fixed architecture, training objective, and seasonal differences are possible factors, not tested explanations.</p>
{image('model_comparison.png','Test Macro-MAE, percentage points; lower is better.')}
{image('test_timeseries.png','True and predicted weighted ILI by target week ending date; all ten regions.')}
{image('predicted_vs_observed.png','Shared axes and y=x line expose bias; this is not complete probabilistic calibration.')}
{image('lstm_training.png','Training loss and validation MAE; dashed line marks restored weights.')}
<h2>Regional and seasonal results</h2>{table(m[m.scope=='season'])}<details><summary>All regional metrics</summary>{table(m[m.scope=='region'])}</details>
<h2>Checks and limitations</h2><p>Automated checks cover CDC week 53, horizon, boundary membership, legal history crossing, bad panel rejection, train-only scaling, prediction keys, manual metrics, CSV metric reconstruction, label perturbation, and model reload agreement (absolute tolerance 1e−6). See run_summary.json and execution_log.txt for actual evidence.</p>
<ul><li>Historical values are revisions retrieved now; this is not a real-time vintage backtest. The horizon begins at the last observed week, not a verified publication date.</li><li>Regions and adjacent windows are dependent. No window-independent significance tests were performed.</li><li>Only 2010–2019, one split, and one seed are examined; conclusions need not apply after the pandemic.</li><li>This is continuous point forecasting. AUROC/AUPRC are inapplicable; course discrimination/calibration requirements still need an appropriate study design, without arbitrary dichotomization.</li><li>No prediction intervals, peak alerts, extra covariates, or additional hyperparameter searches were introduced.</li></ul>
<h2>Reproduction and assistance</h2><p>Use the included locked requirements, raw snapshot and README commands. Models are saved locally. Codex implemented, debugged, ran, checked and documented this demo; record that assistance according to course rules if used in a course project.</p>
<p>Next discussion: real-time data availability and vintage design; an appropriate regression discrimination/calibration design; robustness across seasons or seeds under a new pre-specified protocol.</p>
<h2>Sources</h2><ul>
<li><a href="https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html">Delphi ILINet definitions, revisions, and public-domain license</a></li>
<li><a href="https://cmu-delphi.github.io/delphi-epidata/api/fluview.html">Delphi legacy FluView endpoint</a></li>
<li><a href="https://cdcepi.github.io/flu/data.html">CDC Epidemic Prediction Initiative historical releases</a></li>
<li><a href="https://epiweeks.readthedocs.io/en/stable/usage.html">EpiWeeks CDC calendar</a></li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html">Scikit-learn gradient boosting</a></li>
<li><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html">PyTorch LSTM</a></li></ul></html>'''
    (out/'demo_report.html').write_text(body)

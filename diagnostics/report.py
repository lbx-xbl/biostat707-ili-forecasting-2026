"""Self-contained report from fixed forecast diagnostics, never refits forecasters."""
from pathlib import Path
import base64
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'Persistence':'#78838b','Seasonal':'#d18b21','Ridge':'#2469ae','HGB':'#39834f','LSTM':'#a04d96'}

def render(out):
    out=Path(out); cfg=json.loads((out/'config_frozen.json').read_text()); models=cfg['models']; primary=cfg['primary_block_length']
    b=pd.read_csv(out/'bootstrap_intervals.csv'); c=pd.read_csv(out/'calibration_components.csv'); points=pd.read_csv(out/'calibration_diagnostic_points.csv')
    figures=out/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric in zip(axes,['MAE','RMSE']):
        g=b[(b.scope=='overall')&(b.block_length==primary)&(b.metric==metric)].set_index('model')
        for i,model in enumerate(models[1:]):
            r=g.loc[model]; ax.hlines(i,r.ci_lower,r.ci_upper,color=COLORS[model],lw=3); ax.scatter(r.estimate,i,color=COLORS[model],s=45,zorder=3)
            ax.text(r.ci_upper+.008,i,f'[{r.ci_lower:.3f}, {r.ci_upper:.3f}]',va='center',fontsize=9)
        ax.axvline(0,color='black',ls='--'); ax.set_yticks(range(4),models[1:]); ax.invert_yaxis(); ax.set_xlabel(f'{metric} difference (percentage points)'); ax.set_title(f'{metric}: model minus Persistence')
        lo,hi=ax.get_xlim(); ax.set_xlim(lo,hi+(hi-lo)*.55); ax.grid(axis='x',alpha=.15)
    fig.suptitle(f'Paired moving-block bootstrap: {primary}-week blocks, 95% percentile intervals')
    fig.savefig(figures/'bootstrap_overall.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    lengths=sorted(b.block_length.unique()); offsets=np.linspace(-.21,.21,len(lengths)); markers=['s','o','^']
    for ax,metric in zip(axes,['MAE','RMSE']):
        for li,length in enumerate(lengths):
            g=b[(b.scope=='overall')&(b.block_length==length)&(b.metric==metric)].set_index('model')
            for i,model in enumerate(models[1:]):
                r=g.loc[model]; yy=i+offsets[li]
                ax.hlines(yy,r.ci_lower,r.ci_upper,color=COLORS[model],lw=1.5)
                ax.scatter(r.estimate,yy,marker=markers[li],color=COLORS[model],s=30,label=f'L={length}' if i==0 else None)
        ax.axvline(0,color='black',ls='--'); ax.set_yticks(range(4),models[1:]); ax.invert_yaxis(); ax.set_xlabel(f'{metric} difference (pp)'); ax.legend(title='Block weeks'); ax.grid(axis='x',alpha=.15)
    fig.suptitle('Predeclared block-length sensitivity: identical point estimates, different intervals')
    fig.savefig(figures/'block_sensitivity.png',dpi=150); plt.close(fig)
    pooled=points[points.scope=='overall']; maximum=float(max(pooled.y_true.max(),pooled.y_pred.max(),pooled.isotonic_diagnostic_mean.max()))*1.03
    minimum=min(0,float(pooled.y_pred.min()))
    def curve(ax,g,model,label=True):
        u=g.sort_values('y_pred').drop_duplicates('y_pred')
        ax.plot(u.y_pred,u.isotonic_diagnostic_mean,color=COLORS[model],lw=1.5,label=model if label else None)
    fig,axes=plt.subplots(2,3,figsize=(14,9),sharex=True,sharey=True,layout='constrained')
    for model,ax in zip(models,axes.flat):
        g=pooled[pooled.model==model]; ax.scatter(g.y_pred,g.y_true,s=4,alpha=.12,color='#66737b')
        curve(ax,g,model,False); ax.plot([minimum,maximum],[minimum,maximum],'k--',lw=1)
        ax.plot(g.y_pred,np.repeat(minimum,len(g)),'|',color=COLORS[model],alpha=.12,markersize=4)
        ax.set(title=model,xlabel='Saved forecast (%)',ylabel='Observed / isotonic mean ILI (%)',xlim=(minimum,maximum),ylim=(minimum,maximum)); ax.set_aspect('equal')
    axes.flat[-1].axis('off'); axes.flat[-1].text(.05,.9,'Mean reliability diagnostic\n\nColor: isotonic conditional-mean fit\nGray: original outcomes\nDashed: y = x\nRug: forecast support\n\nSame evaluation data used to fit curve.\nNot an independent performance gain.\nNo confidence bands or calibration tests.',transform=axes.flat[-1].transAxes,va='top',fontsize=11)
    fig.suptitle('Pooled CORP mean reliability: all six seasons and ten regions')
    fig.savefig(figures/'reliability_overall.png',dpi=150); plt.close(fig)
    for scope,groups,shape,size,name in [('season',cfg['seasons'],(2,3),(14,9),'reliability_seasons.png'),('region',cfg['regions'],(5,2),(13,21),'reliability_regions.png')]:
        fig,axes=plt.subplots(*shape,figsize=size,sharex=True,sharey=True,layout='constrained')
        for group,ax in zip(groups,axes.flat):
            for model in models: curve(ax,points[(points.scope==scope)&(points.group==group)&(points.model==model)],model)
            ax.plot([minimum,maximum],[minimum,maximum],'k--',lw=.8)
            ax.set(title=group,xlabel='Saved forecast (%)',ylabel='Isotonic mean ILI (%)',xlim=(minimum,maximum),ylim=(minimum,maximum)); ax.set_aspect('equal')
        handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='outside upper center',ncol=5)
        fig.savefig(figures/name,dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    cc=c[c.scope=='overall'].set_index('model').loc[models]
    for model in models:
        row=cc.loc[model]; axes[0].scatter(row.mcb,row.dsc,color=COLORS[model],s=65)
        axes[0].annotate(model,(row.mcb,row.dsc),xytext=({'HGB':(9,12),'LSTM':(9,-12)}.get(model,(6,5))),textcoords='offset points')
    axes[0].set(xlabel='Miscalibration MCB (pp²; lower is better)',ylabel='Discrimination DSC (pp²; higher is better)',title='Pooled score components'); axes[0].margins(.25)
    axes[1].bar(models,cc.mcb_unconditional,color='#bdc6ce',label='Unconditional: squared bias')
    axes[1].bar(models,cc.mcb_conditional,bottom=cc.mcb_unconditional,color=[COLORS[m] for m in models],label='Remaining conditional component')
    axes[1].set(ylabel='Miscalibration (squared percentage points)',title='MCB split: average bias and conditional shape'); axes[1].tick_params(axis='x',rotation=25); axes[1].legend(fontsize=9)
    fig.savefig(figures/'decomposition.png',dpi=150); plt.close(fig)
    inclusion=pd.read_csv(out/'bootstrap_week_inclusion.csv')
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for length in lengths:
        g=inclusion[(inclusion.season==cfg['seasons'][0])&(inclusion.block_length==length)]
        ax.plot(np.arange(len(g))+1,g.mean_copies_per_replicate,label=f'L={length}')
    ax.axhline(1,color='black',ls='--',lw=1); ax.set(xlabel='Eligible week position (first outer season)',ylabel='Mean copies per bootstrap replicate',title='Noncircular blocks underweight edge weeks: an explicit limitation'); ax.legend()
    fig.savefig(figures/'bootstrap_edge_weights.png',dpi=150); plt.close(fig)
    def table(df): return df.to_html(index=False,border=0,float_format=lambda v:f'{v:.5f}',na_rep='NA')
    def img(name,caption):
        encoded=base64.b64encode((figures/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    principal=b[(b.scope=='overall')&(b.block_length==primary)]
    brief=principal[['model','metric','estimate','ci_lower','ci_upper','bootstrap_center_shift']]
    components=c[c.scope=='overall'][['model','mse','mcb','dsc','unc','root_mcb','bias','mcb_unconditional','mcb_conditional','skill_vs_evaluation_mean']]
    reference='https://doi.org/10.1214/23-EJS2180'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ILI: Frozen-Forecast Uncertainty and Calibration</title>
<style>body{{max-width:1200px;margin:38px auto;padding:0 24px;font:16px/1.6 system-ui;color:#19242d}}h1{{line-height:1.2}}h2{{margin-top:36px}}table{{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#edf3f8}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#52616d}}.note{{padding:16px;background:#edf4f9}}code{{background:#f3f5f6}}</style>
<h1>Regional ILI: Frozen-Forecast Uncertainty and Mean Calibration</h1>
<p class="note">v0.3, phase two. Five frozen methods, 3,010 shared outer-test samples per method, six seasons (2013–14 to 2018–19). No forecasting model was retrained and no forecast replaced. The original pilot and rolling phase-one artifacts remain byte-identical.</p>
<h2>Questions and estimands</h2><p>How uncertain are MAE and RMSE differences versus Persistence within the observed seasons? How do existing point forecasts behave as conditional-mean forecasts? These are separate questions: bootstrap comparisons assess original loss differences; isotonic fits diagnose the relationship between forecasts and outcomes. Neither constitutes a new forecasting experiment.</p>
<p>ILI is weighted outpatient ILI percentage. MAE/RMSE/bias and differences use percentage points (pp), while MSE/MCB/DSC/UNC use squared percentage points (pp²). Macro-MAE weights ten regions equally; RMSE pools squared errors before taking the square root. Fixed complete panels give equal observation counts per region. A 53-week season contributes one additional eligible week.</p>
<h2>Paired moving-block bootstrap</h2><p>The protocol fixes {cfg['bootstrap_replicates']:,} replicates, seed {cfg['seed']}, primary length {primary} weeks, and sensitivity lengths 4 and 13 before the diagnostic run. For each season, sample overlapping blocks uniformly from valid starts, concatenate and truncate to its original 50 or 51 eligible weeks. Never wrap or cross seasonal boundaries. All ten regions and all five models use identical sampled weeks. The same draws support every metric and subgroup. Regions and seasons themselves are not resampled.</p>
<p>Difference = model minus Persistence, so negative favors the model. Points are original differences; intervals are marginal 95% percentile intervals from paired replicates. They are not multiplicity-adjusted tests or model-winning probabilities. Bootstrap centering shift is the replicate mean minus the original estimate. <a href="https://doi.org/10.1214/aos/1176347265">Künsch (1989)</a> provides the stationary dependent-data motivation; our seasonal application is approximate.</p>
<p><strong>Observed findings:</strong> all three learned models have lower original MAE than Persistence, with eight-week intervals entirely below zero. Only Ridge has an RMSE interval entirely below zero; HGB and LSTM intervals include zero. Seasonal is worse on both metrics. These are comparisons against Persistence, not pairwise comparisons among learned models.</p>
{table(brief)}
{img('bootstrap_overall.png','Overall differences versus Persistence, with primary eight-week percentile intervals. Zero is a reference, not a formal multiplicity-adjusted testing rule.')}
{img('block_sensitivity.png','All predeclared block lengths are shown. Do not select a length because its interval favors a model.')}
<p>The MAE interval directions and Ridge/HGB RMSE interpretations persist at all three block lengths. LSTM RMSE includes zero at lengths 4 and 8 but excludes zero at length 13; its interval interpretation is sensitive to block length, so the predeclared eight-week result remains primary.</p><p>Subgroup intervals are exploratory. The bootstrap holds the fitted models, observed regions, historical revisions and seasons fixed; it excludes training/selection uncertainty and does not establish rankings in new seasons. Shared training across folds and dependence beyond block length are not fully represented. Strong seasonal nonstationarity challenges stationary-block assumptions, so nominal coverage is not guaranteed.</p>
{img('bootstrap_edge_weights.png','Ordinary noncircular blocks weight interior weeks more heavily. This can shift bootstrap centering; intervals are conditional approximate summaries, not general confidence guarantees.')}
<details><summary>All overall sensitivity intervals</summary>{table(b[b.scope=='overall'])}</details>
<details><summary>Seasonal primary intervals</summary>{table(b[(b.scope=='season')&(b.block_length==primary)])}</details>
<details><summary>Regional primary intervals</summary>{table(b[(b.scope=='region')&(b.block_length==primary)])}</details>
<h2>Conditional-mean reliability and score decomposition</h2><p>Following <a href="{reference}">Gneiting and Resin (2023), section 3.3</a>, use nondecreasing isotonic least squares, with equal weights and tied forecasts sharing fitted means. Let S be original MSE, S_iso diagnostic isotonic MSE, and UNC the population-denominator variance of evaluation outcomes. Define MCB = S − S_iso and DSC = UNC − S_iso, giving S = MCB − DSC + UNC. Unconditional MCB is squared mean bias; the conditional remainder is MCB minus squared bias. This empirical score-difference formula is not generally the mean squared distance between raw and isotonic predictions.</p>
<p>MCB quantifies score improvement attainable by this same-sample monotone adjustment; DSC describes separation of outcome levels after that adjustment. UNC depends only on outcomes. Skill 1 − S/UNC compares to the evaluation group's empirical mean and may be negative; if UNC is zero it is NA. That mean is a diagnostic reference, not an available forecasting baseline.</p>
<p><strong>Observed decomposition:</strong> Ridge has the smallest pooled MCB (0.05189 pp²), while HGB has the largest DSC (2.23615 pp²). Relative to Ridge, HGB gains about 0.02410 pp² in DSC but incurs about 0.03044 pp² more MCB; the net 0.00634 pp² higher MSE explains its slightly worse RMSE despite better MAE. This is an empirical squared-loss explanation, not a causal explanation of model training. Persistence has near-zero average bias but nonzero MCB, illustrating that offsetting conditional errors can hide behind a small overall bias.</p>
{table(components)}
{img('reliability_overall.png','Pooled mean reliability with outcome scatter and forecast-support rug. The isotonic function is fitted to the same evaluation outcomes; it is not a validated new forecasting model.')}
{img('decomposition.png','All plotted components refer to squared-error mean forecasts, not classification discrimination and not a decomposition of MAE or RMSE.')}
<h2>Why show regions and seasons separately?</h2><p>Pooled reliability can hide regional and seasonal differences. Each curve below is fitted within its labeled subgroup. Extreme forecast values have limited support; monotone fits regularize their shape but do not prove population monotonicity. Subgroup component magnitudes have different outcome variances, so compare methods within the same subgroup before comparing across groups.</p>
{img('reliability_seasons.png','Six separate seasonal mean reliability fits; same saved forecasts, no recalibration deployment.')}
{img('reliability_regions.png','Ten separate regional mean reliability fits; all six seasons in each region.')}
<details><summary>All seasonal components</summary>{table(c[c.scope=='season'])}</details>
<details><summary>All regional components</summary>{table(c[c.scope=='region'])}</details>
<h2>Interpretation boundaries</h2><ul><li>Mean calibration corresponds to squared loss; absolute loss is median-optimal. MAE remains the primary accuracy metric, but its decomposition is not inferred from these mean curves.</li><li>Isotonic fits and component reductions are same-sample diagnostics and can be optimistic in finite samples. No recalibrated forecast is substituted, and no calibration hypothesis test or confidence band is claimed.</li><li>No iid resampling or independence-based significance claims are used. The moving-block intervals nevertheless rely on approximate dependence assumptions that are difficult to justify fully for short, seasonal series.</li><li>These are revised historical ILINet data. Earlier years appeared in pilot training; later pilot results and all rolling point metrics were already viewed. New diagnostic rules were fixed before this run, but the work is exploratory retrospective evaluation, not untouched confirmation.</li><li>The two-season pilot is retained separately, not pooled as extra data. Probabilistic forecasts/WIS and as-of vintage reconstruction remain future stages.</li></ul>
<h2>Reproduction, checks and AI use</h2><p>See README_DIAGNOSTICS.md. Run <code>python run_diagnostics.py --config config/diagnostics.json --output-dir outputs/diagnostics_reproduction</code>; use a fresh output directory. The supplied saved predictions are sufficient for these calculations. Config/protocol hashes, preservation manifests, raw bootstrap draw indices, replicate differences and per-sample isotonic diagnostic values support inspection. Checks include paired keys, block contiguity, finite and tied outcomes, decomposition identities, independent recomputation, preservation of prior files and archive extraction.</p><p>Codex implemented, ran, tested and documented this stage and inspected its figures. Team members must supply their own account of what they verified and learned; no personal reflection is invented. Model and analysis code assistance should be disclosed according to course rules.</p>
<h2>References</h2><ul><li>Gneiting, T. and Resin, J. (2023). <a href="{reference}">Regression diagnostics meets forecast evaluation: conditional calibration, reliability diagrams, and coefficient of determination</a>. Electronic Journal of Statistics 17, 3226–3286. Equations 26–31.</li><li>Künsch, H. R. (1989). <a href="https://doi.org/10.1214/aos/1176347265">The Jackknife and the Bootstrap for General Stationary Observations</a>. Annals of Statistics 17, 1217–1241.</li><li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html">scikit-learn IsotonicRegression documentation</a>.</li><li><a href="https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html">CDC ILINet via Delphi: definitions, revisions and public-domain source</a>.</li></ul></html>'''
    (out/'diagnostics_report.html').write_text(body)

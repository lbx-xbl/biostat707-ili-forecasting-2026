"""Render frozen point forecasts only; this module never trains models."""
import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import MODELS

COLORS={'Persistence':'#7b858d','Seasonal':'#d99021','Ridge':'#2469ae','HGB':'#368453','LSTM':'#9c4a98'}
def render(out):
    out=Path(out); figs=out/'figures'; figs.mkdir(exist_ok=True)
    p=pd.read_csv(out/'predictions.csv'); m=pd.read_csv(out/'metrics.csv'); choices=pd.read_csv(out/'model_selection_summary.csv'); splits=pd.read_csv(out/'split_summary.csv')
    seasons=sorted(p.target_season.unique()); overall=m[m.scope=='overall'].set_index('model').loc[MODELS]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for ax,col,label in zip(axes,['macro_mae','pooled_rmse','bias'],['Macro-MAE','Pooled RMSE','Mean signed error']):
        bars=ax.bar(MODELS,overall[col],color=[COLORS[k] for k in MODELS]); ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=9)
        ax.axhline(0,color='#333',lw=.7); ax.set_ylabel('Percentage points'); ax.set_title(label); ax.tick_params(axis='x',rotation=25)
        lo,hi=ax.get_ylim(); ax.set_ylim(lo-(hi-lo)*.08,hi+(hi-lo)*.18)
    fig.suptitle('Six-season rolling evaluation: 2013–14 to 2018–19')
    fig.savefig(figs/'overall_comparison.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True,layout='constrained')
    for ax,col,label in zip(axes,['macro_mae','pooled_rmse','bias'],['Macro-MAE','Pooled RMSE','Mean signed error']):
        for model in MODELS:
            g=m[(m.scope=='season') & (m.model==model)].set_index('group').loc[seasons]
            ax.plot(seasons,g[col],marker='o',color=COLORS[model],label=model)
        ax.set_ylabel(f'{label}\n(percentage points)'); ax.grid(alpha=.2)
    axes[0].legend(ncol=5); axes[-1].tick_params(axis='x',rotation=20)
    fig.savefig(figs/'season_comparison.png',dpi=150); plt.close(fig)
    regions=[f'hhs{i}' for i in range(1,11)]
    fig,axes=plt.subplots(5,2,figsize=(16,15),sharex=True,layout='constrained')
    for r,ax in zip(regions,axes.flat):
        g=p[p.region==r]
        # Draw seasons separately: do not interpolate across excluded boundary origins.
        for j,s in enumerate(seasons):
            sg=g[g.target_season==s]
            obs=sg[sg.model=='Persistence'].sort_values('target_date')
            ax.plot(pd.to_datetime(obs.target_date),obs.y_true,color='#19242d',lw=1.2,label='Observed' if j==0 else None)
            for model in MODELS:
                q=sg[sg.model==model].sort_values('target_date')
                ax.plot(pd.to_datetime(q.target_date),q.y_pred,color=COLORS[model],alpha=.75,lw=.8,label=model if j==0 else None)
        ax.set_title(r.upper()); ax.set_ylabel('Weighted ILI (%)'); ax.grid(alpha=.12)
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='outside upper center',ncol=6)
    fig.supxlabel('Target week ending date'); fig.savefig(figs/'regional_timeseries.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for season,ax in zip(seasons,axes.flat):
        c=pd.read_csv(out/'folds'/season/'selection_curve.csv'); q=choices[choices.fold==season].iloc[0]
        ax.plot(c.epoch,c.validation_macro_mae,color=COLORS['LSTM']); ax.axvline(q.lstm_best_epoch,ls='--',color='black')
        ax.set(title=season,xlabel='Selection epoch',ylabel='Validation Macro-MAE (pp)')
    fig.suptitle('LSTM selection curves; dashed line = epoch count used for refitting')
    fig.savefig(figs/'lstm_selection.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for season,ax in zip(seasons,axes.flat):
        c=pd.read_csv(out/'folds'/season/'refit_curve.csv')
        ax.plot(c.epoch,c.train_mse_standardized,color=COLORS['LSTM']); ax.set(title=season,xlabel='Fixed refit epoch',ylabel='Train MSE (standardized)')
    fig.suptitle('LSTM final refit training loss; no outer-test early stopping')
    fig.savefig(figs/'lstm_refit.png',dpi=150); plt.close(fig)
    pilot=pd.read_csv('outputs/demo_v0_1/metrics.csv'); pilot=pilot[(pilot.scope=='overall') & (pilot.split=='Test')]
    # Report coverage differences explicitly, not as an apparent paired comparison.
    old=pd.read_csv('outputs/demo_v0_1/predictions.csv'); old=old[(old.split=='Test') & (old.model=='Persistence')]
    new=p[(p.target_season.isin(['2017-2018','2018-2019'])) & (p.model=='Persistence')]
    overlap={'pilot_n':len(old),'rolling_last_two_n':len(new),'common_n':len(set(old.sample_id)&set(new.sample_id)),
             'pilot_only_n':len(set(old.sample_id)-set(new.sample_id)),'rolling_only_n':len(set(new.sample_id)-set(old.sample_id))}
    (out/'pilot_overlap.json').write_text(json.dumps(overlap,indent=2))
    def table(d): return d.to_html(index=False,border=0,float_format=lambda v:f'{v:.5f}')
    def img(name,caption):
        b=base64.b64encode((figs/name).read_bytes()).decode()
        return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    season_m=m[m.scope=='season']; winners=season_m.loc[season_m.groupby('group').macro_mae.idxmin(),['group','model','macro_mae']]
    overall_best=overall.macro_mae.idxmin(); neg=int(overall.negative_predictions.sum())
    audit=json.loads((out/'data_audit.json').read_text())
    selected_boundary=choices[choices.ridge_alpha.isin([.1,100000])].fold.tolist()
    wins=winners.model.value_counts().to_dict()
    larger_alpha=int((choices.ridge_alpha>100).sum())
    seasonal_wins=int((season_m[season_m.model=='Seasonal'].set_index('group').macro_mae < season_m[season_m.model=='Persistence'].set_index('group').macro_mae).sum())
    comparison_text=(f"HGB has the lowest seasonal MAE in {wins.get('HGB',0)} of six seasons, and LSTM in {wins.get('LSTM',0)}. "
                     f"The lowest aggregate RMSE belongs to {overall.pooled_rmse.idxmin()}. "
                     f"The seasonal baseline beats Persistence MAE in {seasonal_wins} of six seasons. "
                     f"{larger_alpha} of six folds select Ridge alpha above 100. "
                     "These are descriptive rankings, not evidence of significant differences.")
    refit_text='The first fold selects on 2012–13 after fitting candidates on 2010–11 and 2011–12, then refits on 2010–11 through 2012–13 to predict 2013–14. Each following fold advances one season.'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Regional ILI: Six-Season Rolling Evaluation</title>
<style>body{{max-width:1200px;margin:38px auto;padding:0 24px;font:16px/1.6 system-ui;color:#19242d}}h1{{line-height:1.2}}h2{{margin-top:36px}}table{{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#eef3f6}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#536270}}.note{{padding:16px;background:#edf4f9}}pre{{white-space:pre-wrap;background:#f5f6f7;padding:15px}}</style>
<h1>Regional ILI Forecasting: Six-Season Rolling-Origin Evaluation</h1>
<p class="note">v0.2 phase one: five point-forecast methods, six outer seasons, expanded Ridge grid and a seasonal baseline. Original frozen pilot retained separately. These are retrospective evaluations of revised data, not real-time or fully untouched confirmatory forecasts.</p>
<h2>Question and data</h2><p>How stable are model performance and rankings across influenza seasons? Predict weighted regional ILI two observation weeks ahead from 12 past weeks, region indicators and known target-date annual features. CDC ILINet via CMU Delphi, ten HHS regions, {audit['rows']:,} rows (2010w40–2019w39). The exact pilot snapshot is reused; no new observations or covariates were added. Weighted ILI is an outpatient percentage, not population infection prevalence. Errors are percentage points.</p>
<p>Snapshot SHA-256: <code>{audit['source_snapshot']['sha256']}</code>. Source acquired {audit['source_snapshot']['fetched_at_utc']}; Public Domain US Government. Release/issue metadata remain excluded from predictive features.</p>
<h2>Rolling design and information use</h2><p>{refit_text} The complete immediately preceding season selects parameters. Candidate scalers/weights exclude that season; after selection, a new scaler and models are fitted using all complete pre-test seasons. LSTM refitting uses exactly the selected epoch count with no validation/test stopping. Seed 2026, CPU ≤4 threads.</p>
<p>Both origin and target must belong to the same selection or outer-test interval. History may cross its beginning; each horizon is 14 calendar days, with CDC week53 retained. Final refitting treats the entire pre-test interval continuously, including windows crossing the old training/validation boundary. Later test origins may use already observed earlier test weeks without retraining.</p>
{table(choices)}
<p>Ridge grid: 0.1, 1, 10, 100, 1,000, 10,000, 100,000. Validation Macro-MAE selects alpha; exact ties favor larger alpha. Grid-edge selections: {html.escape(str(selected_boundary))}. HGB and the 16-unit, one-layer LSTM retain all pilot hyperparameters. Full settings and per-fold choices are saved with the artifacts.</p>
<p>Seasonal baseline: the mean for the same region and CDC week across complete earlier seasons, fixed throughout each target season. If week53 has no earlier examples, use historical week52 and week1 observations; otherwise use genuine earlier week53 values. Reference tables record counts, date ranges and fallback methods.</p>
<h2>Overall results</h2><p>Each model is evaluated on {len(p)//5:,} identical region-origin-target samples. Macro-MAE gives equal weight to each region; pooled metrics weight observations equally. Consequently the 53-week season contributes one additional eligible week. Bias = prediction minus observation; negative values indicate underprediction. Also report macro bias, averaging regional mean errors. Relative MAE improvements use Persistence or Seasonal as labeled. No prediction clipping is applied; {neg} negative predictions were observed across models.</p>
{table(m[m.scope=='overall'])}
<p>{overall_best} has the lowest aggregate Macro-MAE in these six folds. Seasonal rankings below show whether that aggregate is representative. {comparison_text} Results are descriptive; no bootstrap intervals, significance tests, calibration decompositions or probabilistic scores have been calculated in phase one.</p>
{img('overall_comparison.png','Aggregate MAE, RMSE and bias on identical outer-test keys.')}
<h2>Season-to-season variation</h2>{table(winners.rename(columns={'group':'season','model':'lowest_MAE_model'}))}
{img('season_comparison.png','Six seasonal results, including the severe 2017–18 season; lower MAE/RMSE is better, bias closer to zero is better.')}
{table(season_m[['group','model','n','macro_mae','pooled_rmse','bias','improvement_pct','improvement_vs_seasonal_pct']])}
<h2>Regional behavior and fitting diagnostics</h2>
{img('regional_timeseries.png','Observed and predicted weighted ILI by target date. Lines are separated across seasonal evaluation boundaries.')}
{img('lstm_selection.png','Validation MAE selects the epoch before final expanded-history fitting.')}
{img('lstm_refit.png','Final LSTM training loss only; neither the loss nor stopping uses outer-test targets.')}
<details><summary>Regional metrics across all outer seasons</summary>{table(m[m.scope=='region'])}</details>
<details><summary>All region × season metrics</summary>{table(m[m.scope=='season_region'])}</details>
<details><summary>Per-region split boundaries and counts</summary>{table(splits)}</details>
<h2>Original pilot, retained as a separate analysis</h2>{table(pilot)}
<p>The pilot used six training seasons, one validation season and a continuous two-season test interval, without final refitting. Its {overlap['pilot_n']} Test samples differ from the {overlap['rolling_last_two_n']} samples in the last two rolling folds; {overlap['common_n']} keys are shared and {overlap['pilot_only_n']} are pilot-only boundary windows. Do not interpret their aggregate metric difference as a paired measure of improvement. The pilot is not a seventh independent outer fold. Original files and original ZIP are checked against their pre-implementation SHA-256 manifest.</p>
<h2>Interpretation, limitations and remaining work</h2><ul>
<li>New rolling predictions do not make all historical data previously unseen: early seasons were pilot training data and the 2017–19 pilot outcomes were viewed. The approved new protocol was frozen before these new fold results, but this remains an expanded retrospective evaluation.</li>
<li>Historical revised ILINet values are used, not as-of releases. The horizon starts at observation week, not a verified publication date. No post-pandemic generalization is established.</li>
<li>Temporal and cross-region dependence mean windows cannot be treated as independent replicates. Six seasons provide more varied situations, not unlimited independent evidence.</li>
<li>Additional data in final refitting can change optimization and the appropriate epoch count; using the validation-selected count is a declared rule, not a guarantee of optimal final fitting.</li>
<li>Next phases: paired moving-block uncertainty and continuous-forecast calibration/discrimination with a predeclared scoring functional; optional quantile/WIS and vintage work. Those analyses have not been performed here.</li></ul>
<h2>Reproduction and AI disclosure</h2><p>Run <code>python run_rolling.py --config config/rolling.json --output-dir outputs/rolling_reproduction</code> in the existing pinned environment. A nonempty output directory is rejected. Use <code>python verify_rolling.py --output-dir outputs/rolling_v0_2</code> for independent artifact checks without retraining. See README_ROLLING.md and run_summary.json for commands, environment, elapsed time and verification results.</p>
<p>Codex implemented and tested the extension, ran the models, generated reports and reviewed artifacts. This enabled automated multi-season evaluation and consistency checks. AI output required methodological and visual review; no statistical understanding or personal reflection is attributed to team members without their input. Each team member should document what they verified, learned and found difficult for the final course disclosure.</p>
<h2>Sources</h2><ul><li><a href="https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html">CDC ILINet via Delphi: definitions, revisions, public-domain license</a></li><li><a href="https://cmu-delphi.github.io/delphi-epidata/api/fluview.html">Delphi FluView API</a></li><li><a href="https://cdcepi.github.io/flu/data.html">CDC historical release information</a></li></ul></html>'''
    (out/'rolling_report.html').write_text(body)

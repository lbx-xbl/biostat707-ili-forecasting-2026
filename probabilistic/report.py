"""English scientific figures and a self-contained bilingual results report."""
from pathlib import Path
import base64,html,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'Persistence':'#78838b','Seasonal':'#d18b21','Ridge':'#2469ae','HGB':'#39834f','LSTM':'#a04d96'}

def render(out):
    out=Path(out);cfg=json.loads((out/'config_frozen.json').read_text());models=cfg['models'];seasons=cfg['seasons'];regions=cfg['regions']
    met=pd.read_csv(out/'metrics.csv');cov=pd.read_csv(out/'coverage.csv');rel=pd.read_csv(out/'quantile_reliability.csv');b=pd.read_csv(out/'wis_bootstrap_intervals.csv');q=pd.read_csv(out/'quantiles_wide.csv');clip=pd.read_csv(out/'clipping_summary.csv')
    figs=out/'figures';figs.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    overall=met[met.scope=='overall'].set_index('model').loc[models];c=cov[cov.scope=='overall'];r=rel[rel.scope=='overall']
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');bottom=np.zeros(len(models))
    for col,label,color in [('dispersion','Weighted interval width','#86aac0'),('overprediction','Below-lower penalty (overprediction)','#d48a54'),('underprediction','Above-upper penalty (underprediction)','#b66170'),('median_component','Weighted median absolute error','#777888')]:
        ax.bar(models,overall[col],bottom=bottom,label=label,color=color);bottom+=overall[col].to_numpy()
    for i,v in enumerate(overall.macro_wis):ax.text(i,v+.006,f'{v:.4f}',ha='center')
    ax.set(ylabel='Macro-WIS (percentage points; lower is better)',title='WIS and its four additive components');ax.margins(y=.30);ax.legend(fontsize=9,loc='upper left',ncol=2)
    fig.savefig(figs/'wis_components.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for model in models:
        g=c[c.model==model].sort_values('nominal_coverage');axes[0].plot(g.nominal_coverage,g.coverage,'o-',label=model,color=COLORS[model]);axes[1].plot(g.nominal_coverage,g.mean_width,'o-',color=COLORS[model],label=model)
    axes[0].plot([0,1],[0,1],'k--',lw=1);axes[0].set(xlabel='Nominal central interval coverage',ylabel='Observed inclusive coverage',xlim=(0,1),ylim=(0,1),title='Marginal coverage');axes[1].set(xlabel='Nominal central interval coverage',ylabel='Mean interval width (pp)',title='Sharpness: read jointly with coverage');axes[0].legend(fontsize=9)
    fig.savefig(figs/'coverage_width.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for model in models:
        a=met[(met.scope=='season')&(met.model==model)].set_index('group').loc[seasons];ax.plot(seasons,a.macro_wis,'o-',label=model,color=COLORS[model])
    ax.set(ylabel='Macro-WIS (pp)',title='Season-specific WIS: same eligible weeks for all methods');ax.legend(ncol=5,fontsize=9);ax.tick_params(axis='x',rotation=15)
    fig.savefig(figs/'wis_seasons.png',dpi=150);plt.close(fig)
    data=met[met.scope=='region'].pivot(index='group',columns='model',values='relative_wis_reduction').loc[regions,models[1:]]*100
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained');limit=max(1,float(np.abs(data.to_numpy()).max()));im=ax.imshow(data,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(4),models[1:]);ax.set_yticks(range(10),regions);ax.set_title('WIS reduction versus probabilistic Persistence (%)')
    for i in range(10):
        for j in range(4):ax.text(j,i,f'{data.iloc[i,j]:.1f}',ha='center',va='center',color='white' if abs(data.iloc[i,j])>.55*limit else 'black')
    fig.colorbar(im,ax=ax,label='Positive = lower WIS');fig.savefig(figs/'wis_regions.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for model in models:
        g=r[r.model==model].sort_values('quantile');ax.plot(g['quantile'],g.observed_cdf,'o-',label=model,color=COLORS[model],markersize=3)
    ax.plot([0,1],[0,1],'k--',lw=1);ax.set(xlabel='Forecast quantile level',ylabel='Empirical fraction: outcome <= quantile',title='Marginal quantile reliability; no iid confidence bands',xlim=(0,1),ylim=(0,1));ax.legend();fig.savefig(figs/'quantile_reliability.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(12,14),sharex=True,layout='constrained')
    for model,ax in zip(models,axes):
        g=q[(q.model==model)&(q.region=='hhs1')&(q.target_season=='2017-2018')].sort_values('target_date');x=pd.to_datetime(g.target_date)
        ax.fill_between(x,g['q_0.025'],g['q_0.975'],alpha=.18,color=COLORS[model],label='95% interval');ax.fill_between(x,g['q_0.250'],g['q_0.750'],alpha=.35,color=COLORS[model],label='50% interval');ax.plot(x,g['q_0.500'],color=COLORS[model],label='Predictive median');ax.plot(x,g.y_true,color='black',lw=1.2,label='Observed');ax.set(title=model,ylabel='ILI (%)')
    axes[0].legend(ncol=4,fontsize=9);axes[-1].set_xlabel('Target week ending');fig.suptitle('Illustration: hhs1, 2017–18\nHistorical revised data; intervals estimated from preceding validation season')
    fig.savefig(figs/'forecast_intervals_hhs1_2017_2018.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for i,model in enumerate(models[1:]):
        for j,length in enumerate(cfg['block_lengths']):
            row=b[(b.scope=='overall')&(b.model==model)&(b.block_length==length)].iloc[0];yy=i+(j-1)*.18
            ax.plot([row.ci_lower,row.ci_upper],[yy,yy],color=COLORS[model],alpha=[1,.65,.4][j],lw=2);ax.scatter(row.estimate,yy,color=COLORS[model],s=25);ax.text(row.ci_upper+.005,yy,f'L={length}',va='center',fontsize=8)
    ax.axvline(0,color='black',ls='--');ax.set_yticks(range(4),models[1:]);ax.invert_yaxis();ax.set(xlabel='WIS difference (pp); negative favors model',title='Paired moving-block 95% percentile intervals; L=8 primary');ax.margins(x=.2)
    fig.savefig(figs/'wis_bootstrap.png',dpi=150);plt.close(fig)
    def table(df):return df.to_html(index=False,border=0,float_format=lambda v:f'{v:.5f}',na_rep='NA')
    def img(name,caption):return f'<figure><img src="data:image/png;base64,{base64.b64encode((figs/name).read_bytes()).decode()}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    featured=c[c.nominal_coverage.isin([.5,.8,.95])]
    concise=overall.reset_index()[['model','macro_wis','relative_wis_reduction','median_mae','median_rmse','median_bias']]
    for level in [.5,.8,.95]:
        vals=featured[featured.nominal_coverage==level].set_index('model');concise[f'coverage_{int(level*100)}']=concise.model.map(vals.coverage);concise[f'width_{int(level*100)}']=concise.model.map(vals.mean_width)
    winner=overall.macro_wis.idxmin();reduction=overall.loc[winner,'relative_wis_reduction']*100
    severe=cov[(cov.scope=='season')&(cov.group=='2017-2018')&(cov.model=='HGB')&(cov.nominal_coverage==.95)].coverage.iloc[0]
    cover95=featured[featured.nominal_coverage==.95].set_index('model').loc[models].coverage
    main=b[(b.scope=='overall')&(b.block_length==8)]
    winners=met[met.scope=='season'].sort_values('macro_wis').groupby('group').first()['model'].reindex(seasons)
    chinese=f'''<h2>中文结果解读</h2><p>本阶段最低总体 WIS 来自 {winner}：{overall.loc[winner,'macro_wis']:.5f}，相对概率 Persistence 降低 {reduction:.1f}%。这是“原点预测模型 + 前一验证季残差分布”的整体表现，不等同于对原点模型的单独排名。</p><p>WIS 同时惩罚区间过宽和漏掉真实值；因此，区间越宽、覆盖率越高并不自动意味着更好。五个方法的 95% 区间实际覆盖率介于 {cover95.min()*100:.1f}% 与 {cover95.max()*100:.1f}%；应结合下表、分季覆盖率和漏报方向解读。预测中位数还包含前一季残差中位数修正，不能把它的 MAE 当成旧点预测的 MAE。</p><p>特别是严重的 2017–18 季，HGB 的 95% 区间覆盖率只有 {severe*100:.1f}%，低于其总体覆盖率。前一季残差无法保证捕捉下一季的极端波动。HGB 的 WIS 领先，也不与之前 Ridge 的点预测 RMSE 最低矛盾：WIS 评价整组分位数在宽度与漏覆盖之间的平衡，RMSE 则对较大的单点误差给予平方惩罚。</p><p>这里的残差来自严格早于预测季的验证季，但该验证季曾用于 Ridge/LSTM 的参数选择；最终点模型又经过扩展训练集重拟合。残差分布跨模型版本、跨季节转移是否成立，需要由覆盖率评价，不能预设。我们没有利用本季结果调节区间，也没有声称独立校准或覆盖率保证。</p>'''
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ILI v0.4: probabilistic forecasts and WIS</title><style>body{{max-width:1200px;margin:35px auto;padding:0 24px;font:16px/1.65 system-ui;color:#20303a}}h1{{line-height:1.2}}h2{{margin-top:34px}}table{{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#edf3f8}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#52616d}}.note{{padding:18px;background:#edf4f9}}</style>
<h1>Regional ILI: Probabilistic Forecasts and Weighted Interval Score</h1><p class="note">v0.4. Six rolling test seasons, 10 HHS regions, 3,010 identical samples per method, five frozen point models, 23 quantiles per forecast. Fit 30 empirical residual distributions; no point-model retraining, new hyperparameter selection or modification of earlier results. This is retrospective evaluation using revised ILINet data.</p>
{chinese}
<h2>Observed overall results</h2>{table(concise)}
<p>WIS, widths and errors use percentage points; coverage and relative reduction in the table are proportions. Macro-WIS averages regional mean scores equally. Every region has the same number of eligible observations, so this also equals pooled mean WIS. Seasonal winners: {html.escape('; '.join(f'{s}: {m}' for s,m in winners.items()))}. This is descriptive ranking, not evidence of universal superiority.</p>
{img('wis_components.png','Four components sum exactly to WIS. The median absolute-error contribution is shown separately from both tail penalties.')}
<h2>What was fitted, and when?</h2><p>For each method and outer season, estimate the 23 empirical quantiles of outcome minus prediction from the immediately preceding validation season, pooling all ten regions. Add these offsets to each frozen refit point forecast and project onto [0,100], the support of a percentage. The linear sample-quantile convention is fixed; no distribution family, scale, residual window or regional grouping is selected on test performance. Quantiles are nondecreasing, including possible ties at zero. Their median may differ from the old point forecast.</p>
<p>All residual target weeks precede the first outer origin week. All six seasons' distributions and predictions were frozen before this run joined evaluation labels. However, Ridge alpha and LSTM stopping epoch were selected using the same validation outcomes that supply residuals. This is selection reuse, not independent calibration. The residual-producing model was trained on less history than the final refit model; error-distribution transport may fail. A common pooled residual shape also ignores heteroscedasticity across region, season phase and predicted level.</p>
<details><summary>Temporal audit</summary>{table(pd.read_csv(out/'temporal_audit.csv'))}</details>
<details><summary>Projection onto [0,100]: all affected quantiles</summary>{table(clip)}</details>
<h2>WIS definition and evaluation</h2><p>Following <a href="https://doi.org/10.1371/journal.pcbi.1008618">Bracher et al. (2021)</a>, for central interval [l,u] with nominal coverage 1−α, IS=(u−l)+(2/α)max(l−y,0)+(2/α)max(y−u,0). With K=11 intervals and median m, WIS=[0.5|y−m|+Σ(α/2)IS]/(K+0.5). Equivalently, WIS is twice the mean pinball loss across all 23 quantiles. Lower is better; it balances sharpness against misses. The saved four components separate width, below-lower penalties, above-upper penalties and the median term. Degenerate distributions reduce to absolute error.</p>
<p>The 23 levels match the widely used <a href="https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md">FluSight quantile format</a>. This project predicts historical regional ILI percentage, not contemporary FluSight hospitalization targets, and is not a FluSight submission. Quantile level selection and WIS scoring do not imply a complete continuous density or a coverage guarantee.</p>
<h2>Coverage, width and reliability</h2>{img('coverage_width.png','Inclusive empirical central-interval coverage and mean width, aggregated with equal region weights. Read together; widening alone is not the objective.')}
{img('quantile_reliability.png','P(outcome <= predicted quantile) versus nominal level. These are marginal diagnostic frequencies; no independent-sample confidence bands or PIT uniformity test is claimed. With tied outcomes or point masses created by support projection, the inclusive frequency need not equal the nominal quantile level even for a valid quantile; gaps alone are not a formal calibration test.')}
<details><summary>All overall interval diagnostics</summary>{table(c)}</details>
<h2>Seasonal and regional heterogeneity</h2>{img('wis_seasons.png','All six seasons; the 53-week season has 51 eligible target weeks, others have 50.')}{img('wis_regions.png','Positive relative reduction favors the method; the probabilistic baseline is Persistence with its own previous-season residual distribution.')}
<details><summary>All season and region scores</summary>{table(met[met.scope!='overall'])}</details>
<details><summary>Seasonal 50%, 80%, 95% coverage and widths</summary>{table(cov[(cov.scope=='season')&cov.nominal_coverage.isin([.5,.8,.95])])}</details>
<details><summary>Regional 50%, 80%, 95% coverage and widths</summary>{table(cov[(cov.scope=='region')&cov.nominal_coverage.isin([.5,.8,.95])])}</details>
{img('forecast_intervals_hhs1_2017_2018.png','An illustration (first HHS region, known severe 2017–18 season), not a representative random sample. All forecasts are indexed by target week.')}
<h2>Conditional uncertainty of WIS differences</h2>{table(main[['model','estimate','ci_lower','ci_upper','bootstrap_center_shift']])}{img('wis_bootstrap.png','5,000 paired season-stratified moving-block replicates, reusing the v0.3 indices. L=8 primary; L=4 and L=13 shown without selecting a preferred result.')}
<p>Difference is method minus probabilistic Persistence. These marginal percentile intervals condition on the observed six seasons, regions, point models and fitted residual distributions. They exclude training, selection and residual-distribution estimation uncertainty. Seasonal nonstationarity, shared training and dependencies longer than the blocks limit their interpretation. Noncircular blocks underweight edge weeks and can shift centering. No multiplicity adjustment, winner probabilities or future-season coverage is claimed. Comparisons against Persistence do not establish pairwise differences among learned methods.</p>
<details><summary>All WIS bootstrap intervals</summary>{table(b)}</details>
<h2>Limitations and AI disclosure</h2><ul><li>Previous test outcomes and point rankings had already been viewed. The new protocol was fixed before WIS computation, but this is exploratory retrospective analysis, not a new untouched confirmation.</li><li>Only one preceding validation season contributes residuals per fold; tail quantiles depend on few, correlated observations. Region-pooled and level-independent spreads can be poorly calibrated locally even when overall coverage looks reasonable.</li><li>Historical revisions and reporting delays are not reconstructed. Two weeks means two observation weeks after the input origin, not a verified two-week real-time release horizon.</li><li>Old pilot, rolling, diagnostics and their archives remain byte-identical. The new median is separate from old point predictions. No calibration diagnostic from v0.3 was used to adjust these distributions.</li><li>Codex designed the residual-wrapper implementation, wrote and ran code/tests, checked numerical identities and prepared this report. AI accelerated implementation and checking but could introduce methodological or coding errors; independent recomputation and review reduce, not eliminate, that risk. Team members must disclose actual use and supply their own account of what they understand and verified.</li><li>Direct quantile gradient boosting and as-of vintage evaluation were not run; the professor's residual-quantile option is the probabilistic method implemented here.</li></ul>
<h2>Reproduce and inspect</h2><p>See README_PROBABILISTIC.md. Use <code>python run_probabilistic.py --config config/probabilistic.json --output-dir outputs/probabilistic_reproduction</code> in a fresh output directory. Then run <code>python verify_probabilistic.py --output-dir outputs/probabilistic_reproduction</code>. Saved source predictions, residuals, quantiles, per-sample interval scores and WIS, grouped tables and bootstrap replicates permit checking without network access or point-model retraining. Keep the archive's relative folder layout for preservation checks.</p>
</html>'''
    (out/'probabilistic_report.html').write_text(body)

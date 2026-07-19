"""Information-dense paper-style figures from completed deep-search cells."""
from pathlib import Path
import re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'outputs/experiments/resonate_and_fire_deep_gain_search';FIG=EXP/'figures';FIG.mkdir(exist_ok=True)
paths=sorted((EXP/'checkpoints').glob('subject_*_classes.npz'));counts=np.array((2,4,8,16,32));subjects=np.arange(1,31)
records=[]
for path in paths:
 m=re.search(r'subject_(\d+)_(\d+)_classes',path.stem);sid=int(m.group(1));count=int(m.group(2))
 with np.load(path) as d:
  target=d['target_scores'];imp=d['best_impostor_scores'];edges=np.histogram_bin_edges(np.r_[target,imp],bins=24);a,_=np.histogram(target,edges,density=True);b,_=np.histogram(imp,edges,density=True);overlap=np.sum(np.minimum(a,b)*np.diff(edges))
  records.append(dict(sid=sid,count=count,accuracy=float(d['accuracy']),alpha=float(d['selected_alpha']),threshold=float(d['selected_threshold']),operating=float(d['selected_operating_rms']),harmonic=int(d['selected_harmonic_index']),spread=int(d['selected_spread_index']),overlap=overlap,margin=np.asarray(d['target_margin']),improvement=float(d['accuracy'])-float(np.max(d['coarse_accuracy'])),coarse=np.asarray(d['coarse_accuracy'])))
complete_subjects=np.array([sid for sid in subjects if sum(r['sid']==sid for r in records)==5]);n=len(complete_subjects)
def matrix(key):return np.array([[next(r[key] for r in records if r['sid']==sid and r['count']==count) for count in counts] for sid in complete_subjects])
acc=matrix('accuracy');alpha=matrix('alpha');threshold=matrix('threshold');operating=matrix('operating');harmonic=matrix('harmonic');spread=matrix('spread');overlap=matrix('overlap');improvement=matrix('improvement')
np.savez_compressed(EXP/'deep_search_plot_snapshot.npz',subject_ids=complete_subjects,class_counts=counts,accuracy=acc,alpha=alpha,threshold=threshold,operating_rms=operating,harmonic_index=harmonic,spread_index=spread,overlap=overlap,expansion_improvement=improvement,completed_cells=len(records))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-ticks');colors=plt.cm.viridis(np.linspace(.12,.88,5))
# Figure 1: performance, variability, overlap, and improvement.
fig,axes=plt.subplots(2,2,figsize=(13,10));ax=axes[0,0]
for i,count in enumerate(counts):
 x=np.full(n,i)+np.linspace(-.12,.12,n);ax.scatter(x,100*acc[:,i],s=24,alpha=.65,color=colors[i]);ax.boxplot(100*acc[:,i],positions=[i],widths=.42,showfliers=False,medianprops={'color':'black'})
ax.plot(np.arange(5),100/counts,'k--o',label='Chance');ax.set(xticks=np.arange(5),xticklabels=counts,xlabel='Classes',ylabel='Apparent accuracy at 1 s (%)',ylim=(0,103));ax.legend(frameon=False)
ax=axes[0,1];im=ax.imshow(100*acc,aspect='auto',cmap='viridis',vmin=0,vmax=100)
ax.set(xticks=np.arange(5),xticklabels=counts,yticks=np.arange(n),yticklabels=[f'S{x}' for x in complete_subjects],xlabel='Classes',ylabel='Subject');fig.colorbar(im,ax=ax,label='Accuracy (%)')
ax=axes[1,0]
for i,count in enumerate(counts):ax.scatter(100*acc[:,i],overlap[:,i],s=35,color=colors[i],alpha=.75,label=str(count))
ax.set(xlabel='Accuracy (%)',ylabel='Target/impostor overlap coefficient',ylim=(0,1.03));ax.legend(title='Classes',frameon=False,ncols=2)
ax=axes[1,1];ax.boxplot([100*improvement[:,i] for i in range(5)],tick_labels=counts,showmeans=True);ax.axhline(0,color='.5',ls='--');ax.set(xlabel='Classes',ylabel='Gain from harmonic/spread refinement (points)')
fig.suptitle(f'Deep-search performance snapshot: {len(records)}/150 cells, {n} complete subjects');fig.tight_layout();fig.savefig(FIG/'09_performance_variability_overlap_overview.png',dpi=180);plt.close(fig)
# Figure 2: selected parameter atlas and effective gain/threshold ratio.
fig,axes=plt.subplots(2,3,figsize=(15,9));items=((alpha,'Damping α','magma',np.array((.005,.01,.025,.05,.1,.2,.4))),(threshold,'Threshold','magma',np.array((.001,.002,.005,.01,.02,.05,.1,.2))),(operating,'Target RMS','viridis',np.array((.1,.25,.5,.75,1.,1.5,2.,3.,5.))),(harmonic,'Harmonic bank','cividis',np.array((0,1,2))),(spread,'Spread bank','cividis',np.array((0,1,2))))
for ax,(values,label,cmap,levels) in zip(axes.ravel()[:5],items):
 indexed=np.searchsorted(levels,values);im=ax.imshow(indexed,aspect='auto',cmap=cmap,vmin=-.5,vmax=len(levels)-.5);ax.set(xticks=np.arange(5),xticklabels=counts,yticks=np.arange(n),yticklabels=[f'S{x}' for x in complete_subjects],xlabel='Classes',ylabel='Subject');ax.set_title(label);bar=fig.colorbar(im,ax=ax,shrink=.75,ticks=np.arange(len(levels)));bar.ax.set_yticklabels([f'{x:g}' for x in levels])
ax=axes.ravel()[5]
for i,count in enumerate(counts):ax.scatter(np.log10(operating[:,i]/threshold[:,i]),100*acc[:,i],color=colors[i],alpha=.75,label=str(count))
ax.set(xlabel='log10(target RMS / threshold)',ylabel='Accuracy (%)');ax.legend(title='Classes',frameon=False,ncols=2)
fig.suptitle('Subject-specific operating-point atlas');fig.tight_layout();fig.savefig(FIG/'10_parameter_selection_atlas.png',dpi=180);plt.close(fig)
# Figure 3: median alpha-threshold landscapes, maximizing over gain.
fig,axes=plt.subplots(1,5,figsize=(16,4.2),sharex=True,sharey=True);last=None
for ci,count in enumerate(counts):
 landscapes=[]
 for r in records:
  if r['count']==count:landscapes.append(r['coarse'].reshape(7,8,9).max(2))
 z=100*np.median(landscapes,axis=0);last=axes[ci].imshow(z,origin='lower',aspect='auto',cmap='magma',vmin=0,vmax=100);axes[ci].set_title(f'{count} classes');axes[ci].set_xticks(np.arange(8),('.001','.002','.005','.01','.02','.05','.1','.2'),rotation=55);axes[ci].set_yticks(np.arange(7),('.005','.01','.025','.05','.1','.2','.4'));axes[ci].set_xlabel('Threshold')
axes[0].set_ylabel('Damping α');fig.colorbar(last,ax=axes.tolist(),label='Median best-gain accuracy (%)',shrink=.8);fig.savefig(FIG/'11_alpha_threshold_landscape_by_class.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Figure 4: raw amplitude phenotype versus behavior and parameters.
profile=np.load(ROOT/'outputs/experiments/resonate_and_fire_gain_optimization/raw_amplitude_and_resonance_profile.npz');amp=profile['subject_median_rms_uV'][complete_subjects-1]
fig,axes=plt.subplots(2,2,figsize=(12,9));metrics=((100*acc.mean(1),'Mean accuracy (%)'),(np.median(alpha,1),'Median damping α'),(np.median(threshold,1),'Median threshold'),(np.median(operating,1),'Median target RMS'))
for ax,(y,label) in zip(axes.ravel(),metrics):
 ax.scatter(amp,y,s=50);label_indices=np.unique(np.r_[np.argsort(y)[:2],np.argsort(y)[-2:],np.argsort(amp)[:1],np.argsort(amp)[-1:]]);[ax.annotate(f'S{complete_subjects[i]}',(amp[i],y[i]),xytext=(3,3),textcoords='offset points',fontsize=8) for i in label_indices];ax.set(xlabel='Median raw RMS (µV)',ylabel=label);ax.grid(alpha=.2)
fig.suptitle('Subject amplitude phenotype and fitted operating point');fig.tight_layout();fig.savefig(FIG/'12_subject_amplitude_parameter_phenotypes.png',dpi=180);plt.close(fig)
print('snapshot',len(records),'cells',n,'complete subjects',FIG)

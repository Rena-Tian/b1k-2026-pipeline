import pyarrow.parquet as pq, glob, sys, numpy as np, pandas as pd
for root in sys.argv[1:]:
    files=sorted(glob.glob(root+'/data/chunk-*/file-*.parquet'))
    rows=[]
    for f in files:
        t=pq.read_table(f, columns=['index','episode_index','task_index','timestamp']).to_pandas()
        idx=t['index'].values
        rows.append(dict(file=f.split('/')[-2]+'/'+f.split('/')[-1], n=len(t), idx0=int(idx[0]), idx1=int(idx[-1]), step1=bool(np.all(np.diff(idx)==1)),
                         eps=sorted(t.episode_index.unique().tolist()), tasks=sorted(t.task_index.unique().tolist()),
                         ep_ts0=bool((t.groupby('episode_index').timestamp.min()==0).all())))
    meta=pd.concat([pd.read_parquet(x, columns=['episode_index','dataset_from_index','dataset_to_index','length']) for x in sorted(glob.glob(root+'/meta/episodes/**/*.parquet', recursive=True))])
    print(root, 'meta episodes', len(meta), 'tasks in data', sorted({x for r in rows for x in r['tasks']}))
    for r in rows: print('  ', r['file'], 'n', r['n'], 'index', r['idx0'], '..', r['idx1'], 'contiguous', r['step1'], 'ts0', r['ep_ts0'], 'eps', r['eps'][0], '..', r['eps'][-1], len(r['eps']))
    # consistency: for each episode, data index range == meta from/to
    bad=0
    for f in files:
        t=pq.read_table(f, columns=['index','episode_index']).to_pandas()
        g=t.groupby('episode_index')['index'].agg(['min','max','count'])
        for ep,row in g.iterrows():
            m=meta[meta.episode_index==ep]
            if len(m)!=1 or int(m.dataset_from_index.iloc[0])!=row['min'] or int(m.dataset_to_index.iloc[0])!=row['max']+1 or int(m.length.iloc[0])!=row['count']: bad+=1
    print('   episodes whose data index range mismatches meta from/to/length:', bad)
    print('   videos:', sorted(glob.glob(root+'/videos/*')))

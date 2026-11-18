"""Fetch official vintage records with system TLS trust; never disable verification."""
import concurrent.futures,gzip,hashlib,json,subprocess,urllib.parse
from pathlib import Path
from datetime import datetime,timezone
BASE='https://api.delphi.cmu.edu/epidata/fluview/'
def fetch(region):
    path=Path(f'data/vintage/raw/{region}.json.gz');meta=path.with_suffix('.meta.json')
    if path.exists() and meta.exists():return json.loads(meta.read_text())
    params={'regions':region,'epiweeks':'201040-201939','issues':'201001-201952'}
    url=BASE+'?'+urllib.parse.urlencode(params)
    r=subprocess.run(['/usr/bin/curl','--fail','--silent','--show-error','--retry','2','--max-time','60','--write-out','\n%{http_code}',url],capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    body,status=r.stdout.rsplit(b'\n',1);data=json.loads(body)
    if data.get('result') not in [1,-2]:raise ValueError(f'Incomplete or invalid response: {region} {data.get("result")}')
    path.write_bytes(gzip.compress(body,mtime=0))
    m={'url':url,'fetched_utc':datetime.now(timezone.utc).isoformat(),'http_status':int(status),'api_result':data['result'],'rows':len(data.get('epidata',[])),'uncompressed_bytes':len(body),'raw_sha256':hashlib.sha256(body).hexdigest(),'gzip_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'file':str(path)}
    meta.write_text(json.dumps(m,indent=2));print(region,m['rows'],flush=True);return m
def normalize(destination='data/vintage/records.csv.gz'):
    import pandas as pd
    from epiweeks import Week
    rows=[r for p in sorted(Path('data/vintage/raw').glob('*.json.gz')) for r in json.loads(gzip.decompress(p.read_bytes())).get('epidata',[])]
    d=pd.DataFrame(rows)[['region','epiweek','issue','release_date','wili']]
    if d.duplicated(['region','epiweek','issue']).any():raise ValueError('Duplicate vintage keys')
    d['week_end']=d.epiweek.map(lambda w:Week.fromstring(str(w)).enddate().isoformat())
    d['release_lag_days']=(pd.to_datetime(d.release_date)-pd.to_datetime(d.week_end)).dt.days
    d.to_csv(destination,index=False,compression={'method':'gzip','mtime':0})

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(fetch,[f'hhs{i}' for i in range(1,11)]))
    Path('data/vintage/download_manifest.json').write_text(json.dumps(results,indent=2))

    normalize()

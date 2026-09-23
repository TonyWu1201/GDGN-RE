import csv,time
from program.acquire.download_file import download
with open('data/canonical/compound_official_key_audit.csv',encoding='utf-8') as f:keys=sorted({r['official_inchikey'] for r in csv.DictReader(f) if 'conflict' in r['status']})
for key in keys:
 url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{key}/property/IsomericSMILES,InChIKey,InChI/JSON'
 download(url,f'data/raw/PubChem/snapshot_20260923/{key}.json','pubchem','snapshot_20260923');time.sleep(.4)

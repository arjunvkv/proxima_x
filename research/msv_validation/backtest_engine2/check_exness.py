"""Check Exness tick data archive for availability."""
import requests, re
from urllib.parse import urljoin

headers = {'User-Agent': 'Mozilla/5.0'}
base = 'https://ticks.ex2archive.com/'

# Check root for directory listing
r = requests.get(base, headers=headers, timeout=10)
print('Root page length:', len(r.text))

# Extract links from the page
links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', r.text)
print('Links found:', links[:30])
print()

# Check if there's a different path pattern or API endpoint
paths = ['api/list', 'api/files', 'data', 'download', 'files', 'list', 'sitemap.xml']
for path in paths:
    url = urljoin(base, path)
    try:
        r2 = requests.get(url, headers=headers, timeout=10)
        content_preview = r2.text[:200].replace('\n', ' ') if len(r2.text) > 50 else r2.text
        print(f'{url}: {r2.status_code} ({len(r2.text)} bytes) -> {content_preview}')
    except Exception as e:
        print(f'{url}: {e}')

# Check nginx autoindex
# S3-style listing
for path in ['?prefix=2025-12/', '?list-type=2', '?delimiter=/']:
    url = base + path
    try:
        r3 = requests.get(url, headers=headers, timeout=10)
        print(f'{url}: {r3.status_code} ({len(r3.text)} bytes)')
        if 'Contents' in r3.text or 'Key' in r3.text or 'CommonPrefixes' in r3.text:
            print(f'  XML listing found!')
            print(f'  {r3.text[:500]}')
    except Exception as e:
        print(f'{url}: {e}')

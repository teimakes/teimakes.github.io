"""Validate published HTML, sitemap, and feeds together after `astro build`."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'dist'
SITE = 'https://teimakes.github.io'

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.lang = None
        self.canonicals, self.alternates, self.schemas, self.hrefs = [], {}, [], []
        self.description = None
        self.h1 = 0
        self.noindex = False
        self.in_schema = False
        self.schema_text = ''
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'html': self.lang = a.get('lang')
        if tag == 'h1': self.h1 += 1
        if tag == 'meta' and a.get('name') == 'robots': self.noindex = 'noindex' in a.get('content', '')
        if tag == 'meta' and a.get('name') == 'description': self.description = a.get('content')
        if tag == 'link' and a.get('rel') == 'canonical': self.canonicals.append(a['href'])
        if tag == 'link' and a.get('hreflang'): self.alternates[a['hreflang']] = a['href']
        if tag == 'a' and a.get('href'): self.hrefs.append(a['href'])
        if tag == 'script' and a.get('type') == 'application/ld+json':
            self.in_schema, self.schema_text = True, ''
    def handle_data(self, data):
        if self.in_schema: self.schema_text += data
    def handle_endtag(self, tag):
        if tag == 'script' and self.in_schema:
            self.schemas.append(json.loads(self.schema_text))
            self.in_schema = False

pages = {}
for file in ROOT.rglob('index.html'):
    route = '/' + str(file.relative_to(ROOT)).removesuffix('index.html')
    page = Page(file.read_text())
    pages[SITE + route] = page
    assert len(page.canonicals) == 1, (route, 'one canonical required')
    expected = route
    if route.startswith('/notebooks/'):
        expected = route.replace('/notebooks/', '/posts/').replace('-full/', '/').replace('-centered/', '/')
    assert page.canonicals == [SITE + expected], (route, page.canonicals)
    assert page.lang == ('ko' if route.endswith('-ko/') else 'en'), route
    assert page.description, route
    if not route.startswith('/notebooks/'): assert page.h1 == 1, (route, page.h1)
    if route.startswith('/posts/'):
        article = next(s for s in page.schemas if s['@type'] == 'BlogPosting')
        assert article['url'] == SITE + route and article['inLanguage'] == page.lang
        assert article['datePublished'] and article['author']['url'] == SITE + '/about/'
        assert '/about/' in page.hrefs, 'schema author must also be visible'
    for href in page.hrefs:
        if href.startswith('/') and not href.startswith('//'):
            path = urlsplit(href).path
            if path and '.' not in path.rsplit('/', 1)[-1]:
                assert path.endswith('/'), (route, 'noncanonical link', href)

for url, page in pages.items():
    assert page.canonicals[0] in pages, (url, 'canonical target missing')
    for lang, alternate in page.alternates.items():
        assert alternate in pages and pages[alternate].lang == lang, (url, alternate)
        assert pages[alternate].alternates == page.alternates, (url, 'non-reciprocal hreflang')

ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
locs = [e.text for e in ET.parse(ROOT / 'sitemap.xml').findall('.//s:loc', ns)]
assert len(locs) == len(set(locs))
assert set(locs) == {url for url, page in pages.items() if page.canonicals == [url] and not page.noindex}, 'sitemap must contain exactly canonical pages'
posts = {url for url in locs if '/posts/' in url}
rss = ET.parse(ROOT / 'rss.xml').findall('./channel/item')
assert {item.findtext('link') for item in rss} == posts
for item in rss:
    body = item.findtext('{http://purl.org/rss/1.0/modules/content/}encoded')
    assert body and '<' in body and 'href="/' not in body and 'src="/' not in body
feed = json.loads((ROOT / 'feed.json').read_text())
assert {item['url'] for item in feed['items']} == posts
assert all(item['content_html'] and item['id'] == item['url'] for item in feed['items'])
assert (ROOT / 'rss.xml').stat().st_size < 10 * 1024 * 1024
assert 'name="msvalidate.01" content="78828D2630C9F5A4081A883DDA4D64EE"' in (ROOT / 'index.html').read_text()
assert 'Sitemap: ' + SITE + '/sitemap.xml' in (ROOT / 'robots.txt').read_text()
assert (ROOT / 'googlea8fa32c94d27f661.html').read_text().strip() == 'google-site-verification: googlea8fa32c94d27f661.html'
for file in ROOT.rglob('*.html'):
    text = file.read_text()
    assert 'listmonk.example.com' not in text and 'mailto:hello@example.com' not in text
    assert 'adsbygoogle.js' not in text, 'ads must not run before account/policy/consent setup'
attention = [page.description for url, page in pages.items() if '/posts/attention-' in url]
assert len(set(attention)) == len(attention), 'series descriptions must be unique'
print(f'SEO checks passed: {len(pages)} HTML pages, {len(locs)} sitemap URLs, {len(posts)} posts, {sum(bool(p.alternates) for p in pages.values()) // 2} translation pairs.')

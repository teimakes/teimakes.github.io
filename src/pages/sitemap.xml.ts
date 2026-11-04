export async function GET(context: { site: URL }) {
  const site = context.site ?? new URL('https://teimakes.github.io/');
  const staticPages = ['', 'about/', 'links/', 'subscribe/', 'privacy/'];
  const postImportResult = import.meta.glob('./posts/*.{md,mdx}', { eager: true });
  const posts = Object.values(postImportResult)
    .filter((p: any) => !p.frontmatter.draft)
    .sort((a: any, b: any) => new Date(b.frontmatter.date) - new Date(a.frontmatter.date));

  const urls = [
    ...staticPages.map((path) => ({ loc: new URL(path, site).href, priority: path === '' ? '1.0' : '0.5' })),
    ...posts.map((p: any) => ({
      loc: new URL(`${p.url.replace(/\/$/, '')}/`, site).href,
      lastmod: new Date(p.frontmatter.date).toISOString(),
      priority: '0.8',
    })),
  ];

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls
  .map(
    (u) => `  <url>
    <loc>${u.loc}</loc>${u.lastmod ? `
    <lastmod>${u.lastmod}</lastmod>` : ''}
    <priority>${u.priority}</priority>
  </url>`
  )
  .join('\n')}
</urlset>
`;

  return new Response(xml, {
    headers: { 'Content-Type': 'application/xml; charset=utf-8' },
  });
}

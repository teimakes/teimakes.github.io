import { canonicalPath } from '../lib/seo';
const xmlEscape = (value: string) => value.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]!));
export async function GET(context: { site: URL }) {
  const site = context.site;
  const staticPages = ['/', '/about/', '/links/', '/subscribe/', '/privacy/'];
  const postImportResult = import.meta.glob('./posts/*.{md,mdx}', { eager: true });
  const posts = Object.values(postImportResult)
    .filter((p: any) => !p.frontmatter.draft)
    .sort((a: any, b: any) => new Date(b.frontmatter.date).getTime() - new Date(a.frontmatter.date).getTime());
  const urls: { loc: string; lastmod?: string }[] = [
    ...staticPages.map((path) => ({ loc: new URL(path, site).href })),
    ...posts.map((p: any) => ({
      loc: new URL(canonicalPath(p.url), site).href,
      lastmod: new Date(p.frontmatter.updatedDate || p.frontmatter.date).toISOString(),
    })),
  ];
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `  <url><loc>${xmlEscape(u.loc)}</loc>${u.lastmod ? `<lastmod>${u.lastmod}</lastmod>` : ''}</url>`).join('\n')}
</urlset>\n`;
  return new Response(xml, { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
}

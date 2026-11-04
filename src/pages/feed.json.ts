import { SITE } from '../config';
import { absoluteFeedHtml, canonicalPath, pageLanguage } from '../lib/seo';

export async function GET(context: { site: URL }) {
  const homePageUrl = context.site.href;
  const postImportResult = import.meta.glob('./posts/*.{md,mdx}', { eager: true });
  const sorted = Object.values(postImportResult)
    .filter((p: any) => !p.frontmatter.draft)
    .sort((a: any, b: any) => new Date(b.frontmatter.date).getTime() - new Date(a.frontmatter.date).getTime());
  const items = await Promise.all(sorted.map(async (p: any) => {
    const url = new URL(canonicalPath(p.url), homePageUrl).href;
    return {
      id: url, url, title: p.frontmatter.title, summary: p.frontmatter.description,
      content_html: p.compiledContent ? absoluteFeedHtml(await p.compiledContent(), url) : p.frontmatter.description,
      date_published: new Date(p.frontmatter.date).toISOString(),
      ...(p.frontmatter.updatedDate ? { date_modified: new Date(p.frontmatter.updatedDate).toISOString() } : {}),
      language: p.frontmatter.lang || pageLanguage(p.url), tags: p.frontmatter.tags,
      authors: [{ name: SITE.author, url: new URL('/about/', homePageUrl).href }],
    };
  }));
  return new Response(JSON.stringify({
    version: 'https://jsonfeed.org/version/1.1', title: SITE.title,
    home_page_url: homePageUrl, feed_url: new URL('feed.json', homePageUrl).href,
    description: SITE.description, items,
  }), { headers: { 'Content-Type': 'application/feed+json; charset=utf-8' } });
}

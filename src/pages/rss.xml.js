import rss from '@astrojs/rss';
import { SITE } from '../config';
import { absoluteFeedHtml, canonicalPath } from '../lib/seo';

export async function GET(context) {
  const postImportResult = import.meta.glob('./posts/*.{md,mdx}', { eager: true });
  const posts = Object.values(postImportResult)
    .filter((p) => !p.frontmatter.draft)
    .sort((a, b) => new Date(b.frontmatter.date) - new Date(a.frontmatter.date));
  return rss({
    title: SITE.title, description: SITE.description, site: context.site,
    items: await Promise.all(posts.map(async (p) => {
      const link = new URL(canonicalPath(p.url), context.site).href;
      return {
        title: p.frontmatter.title, description: p.frontmatter.description,
        pubDate: new Date(p.frontmatter.date), link,
        content: p.compiledContent ? absoluteFeedHtml(await p.compiledContent(), link) : p.frontmatter.description,
      };
    })),
  });
}

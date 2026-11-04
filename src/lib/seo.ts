export function canonicalPath(pathname: string): string {
  const path = pathname.replace(/\/+$/, '') || '/';
  return path === '/' || /\.[a-z0-9]+$/i.test(path) ? path : `${path}/`;
}

export function pageLanguage(pathname: string): 'ko' | 'en' {
  return /-ko\/?$/.test(pathname) ? 'ko' : 'en';
}

// Raw imports avoid a cycle between Markdown pages and their layout.
const postSources = import.meta.glob('../pages/posts/*.{md,mdx}', { eager: true, as: 'raw' });
const publishedPaths = new Set(Object.entries(postSources)
  .filter(([, source]) => !/^draft:\s*true\s*$/m.test(source as string))
  .map(([file]) => canonicalPath(file.replace('../pages', '').replace(/\.(md|mdx)$/, ''))));

export function translations(pathname: string) {
  const path = canonicalPath(pathname);
  if (!publishedPaths.has(path)) return [];
  const english = path.replace(/-ko\/$/, '/');
  const korean = english.replace(/\/$/, '-ko/');
  if (!publishedPaths.has(english) || !publishedPaths.has(korean)) return [];
  return [{ lang: 'en', path: english }, { lang: 'ko', path: korean }];
}

// Feed readers resolve links outside the site's own page context.
export function absoluteFeedHtml(html: string, pageUrl: string): string {
  return html.replace(/\b(href|src)=(['"])([^'"]+)\2/gi, (match, attr, quote, value) => {
    if (/^(?:https?:|mailto:|data:|tel:)/i.test(value)) return match;
    return `${attr}=${quote}${new URL(value, pageUrl).href}${quote}`;
  });
}

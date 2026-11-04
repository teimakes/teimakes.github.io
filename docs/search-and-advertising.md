# Search and advertising operations

## Search registration

Use the HTTPS root `https://teimakes.github.io/` for all properties. Verification tokens are public proofs of site ownership, not account credentials; keep any issued verification files in `public/` after verification.

- Google Search Console: ownership already verified through `googlea8fa32c94d27f661.html`. `sitemap.xml` was submitted; last observed status was "Couldn't fetch". Successful URL live tests and indexing requests do not mean that Google has indexed the pages.
- Bing Webmaster Tools: pending Microsoft profile-access consent before registration. Add the site manually, publish the exact XML verification file or meta tag provided by Bing, verify ownership, then submit `https://teimakes.github.io/sitemap.xml`. An import from Search Console is optional and requires a separate access grant.
- Naver Search Advisor: sign-in blocked after identity verification by a site-reported unstable-network error in the cloud browser; no site registration or verification token obtained. Add the HTTPS root in Webmaster Tools, publish its exact HTML verification file or meta tag, verify ownership, then submit `https://teimakes.github.io/sitemap.xml` under Requests → Sitemap and `https://teimakes.github.io/rss.xml` under Requests → RSS. Request collection for the home page and a few substantial posts.

Check crawl/index reports after several days. Record the status and date rather than treating submission as an indexing guarantee. Search Console URL Inspection is more authoritative for a specific Google URL than a `site:` search. Avoid repeated submission of unchanged URLs.

The wildcard allow rule in `robots.txt` permits search crawlers, including AI search crawlers. Search discoverability and permission for model training are separate choices; do not change training crawler policies without the owner's decision. No special `llms.txt` is required for Google AI search.

## Publishing checks

Run `npm run build && npm run check:seo`. The check validates canonical URLs, language, reciprocal translations, article metadata, sitemap coverage, feed bodies, and verification file preservation. Markdown pages use `Post.astro`; do not add a duplicate catch-all route.

- Use a clear, specific title and a unique description describing the actual post.
- Use `-ko` for Korean slugs and matching English slugs only for genuine translations. The shared head emits reciprocal hreflang only when both published files exist.
- Keep original `date`; add `updatedDate` only after a meaningful content revision. It controls visible revision dates, structured data, feeds, and sitemap lastmod.
- Include original experiments, setup, results, limitations, and primary references when appropriate. Link related posts naturally.
- The generated notebook layouts canonicalize to the Markdown post and stay out of the sitemap.
- `draft: true` removes a page from discovery feeds/lists and adds `noindex`, but the URL is still public. Do not commit private drafts to this public repository.

## Advertising readiness

Advertising is **not enabled**. There is no AdSense account ID, ad slot, ad script, ads.txt claim, or consent service in this change. The live privacy page describes the current site; update it before any tracking or advertising is enabled.

Before activation:

1. The owner selects/creates the AdSense account, reviews the current terms, and supplies the real publisher ID. Submit the site through that account and follow its ownership/review instructions.
2. Review current GitHub Pages terms. Personal blogs with incidental advertising are distinct from sites primarily operating an online business. If monetization becomes the primary purpose, choose a commercial host. A custom domain is optional; purchase and migration require the owner's choice.
3. Add an accurate privacy disclosure for the actual provider and data use, including cookies/device identifiers, purposes, third-party processing, choices, and relevant provider links. Configure any consent requirements before loading tags, including a Google-certified CMP when serving ads in the EEA, UK, or Switzerland.
4. Use only the ads.txt record issued by the account. Do not invent a publisher ID or copy another publisher's record.
5. Start with one responsive, clearly labelled ad after the article body and before the RSS subscription box. Reserve space to reduce layout shift. Keep code, navigation, and article headings clear. Review on a narrow screen before rollout.
6. Verify the consent behavior, ad network requests, ads.txt reachability, policy wording, and page performance. Keep analytics optional and separately documented.

Measure impressions/clicks in search consoles first. Once traffic is established, assess real revenue rather than assuming an earnings level or ad approval threshold.

## Official references

- https://developers.google.com/search/docs/fundamentals/ai-optimization-guide
- https://developers.google.com/search/docs/specialty/international/localized-versions
- https://www.bing.com/webmasters/help/add-and-verify-site-12184f8b
- https://searchadvisor.naver.com/guide/faq-start-register
- https://searchadvisor.naver.com/guide/request-feed
- https://support.google.com/adsense/answer/1348695
- https://support.google.com/adsense/answer/13554116
- https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies

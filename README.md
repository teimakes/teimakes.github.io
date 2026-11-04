# AI Notebook — Personal AI Blog

A minimal, clean personal blog focused on AI, LLMs, AI Agents, AI Movies, AI Animation and AI Manga. Built with [Astro](https://astro.build), posts written in Markdown, hosted at **https://teimakes.github.io/** with GitHub Pages.

## Writing posts

Add a `.md` file to `src/pages/posts/`:

```markdown
---
layout: '../../layouts/Post.astro'
title: 'My Post Title'
description: 'A short excerpt shown on the homepage and in feeds.'
date: '2025-04-01'
tags: ['llm', 'agents']
---

Your content here...
```

Set `draft: true` to omit a post from lists and feeds and add `noindex`. Its URL remains public; never commit private drafts.

## Posting Jupyter notebooks

You can turn a Jupyter notebook into a blog post **without writing any Markdown** — the notebook itself becomes the post, including its code, outputs, and plots.

1. Put your notebook in `notebooks/` (e.g. `notebooks/my-tutorial.ipynb`).
2. Add a **raw cell at the very top** of the notebook containing the post metadata as JSON:

   ```json
   {
     "title": "My Tutorial Title",
     "description": "Short excerpt for the homepage and feeds.",
     "date": "2025-04-01",
     "tags": ["tutorial", "machine-learning"]
   }
   ```

3. Run the conversion script:

   ```bash
   pip install nbconvert jupyter matplotlib  # plus whatever your notebook imports
   python scripts/notebook_to_post.py        # converts all notebooks in notebooks/
   ```

The script **executes the notebook** (so plots/outputs are always fresh) and converts it to Markdown with outputs embedded, copies the generated images to `public/notebook-assets/<notebook-name>/`, and writes the final post to `src/pages/posts/<notebook-name>.md` with front matter prepended. Commit and push — the post goes live with the rest of the blog.

The script also generates **nbconvert HTML notebook versions** of each notebook, embedded as-is on dedicated pages in two layouts so you can compare designs:

* `/notebooks/<notebook-name>-full` — notebook content spans the full screen width (like the [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html) pages)
* `/notebooks/<notebook-name>-centered` — notebook content is centered and occupies a fixed max width (920px)

The classic Markdown post remains available at `/posts/<notebook-name>`.

The converted `.md` and assets are checked in, so the GitHub Pages build never executes notebooks. Its SEO validation uses Python's standard library.

## Development

```bash
npm install
npm run dev       # local dev server
npm run build     # production build to ./dist
npm run check:seo # validate generated pages, sitemap and feeds
npm run preview   # preview the production build
```

## Deploying to GitHub Pages

The repository is `teimakes/teimakes.github.io`, and the site is published at https://teimakes.github.io/.

1. In the repository settings, open **Pages** and set **Source** to **GitHub Actions**.
2. Push changes to `main`. The workflow in `.github/workflows/deploy.yml` installs dependencies, builds the site, and deploys `dist` to GitHub Pages.
3. You can also start the **Deploy to GitHub Pages** workflow manually from the Actions tab.

The Astro `base` is `/`. Keep internal links and notebook asset paths relative to the domain root.

## Configuration

Edit `src/config.ts` to change the site title, author, description, social links, and subscription URL. Update `site` in `astro.config.mjs` to your final domain.

## Subscriptions and search

RSS and JSON feeds with article bodies are available at `/rss.xml` and `/feed.json`. `/subscribe/` explains how to use them; no email service is connected.

See [search and advertising operations](docs/search-and-advertising.md) for registration status, publishing conventions, and the steps needed before enabling advertising.

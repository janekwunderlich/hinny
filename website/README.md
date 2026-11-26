# Hinny's website

The site for [hinny.dev](https://hinny.dev). TanStack Start (React 19, Vite) with TanStack Router
for the pages, [StyleX](https://stylexjs.com) for every style and
[React Aria Components](https://react-spectrum.adobe.com/react-aria/) for everything interactive.
Every page is prerendered to HTML at build time; [Nitro](https://nitro.build) builds the output,
in Vercel's format when it builds on Vercel.

| Page | |
|---|---|
| `/` | what Hinny is in one line, the kinds of search, the three ways to use it (Rust, Python, C), measured numbers with their conditions, the name, the latest news, where to start |
| `/news` | the latest post large, then every post as a row |
| `/news/<slug>` | a post: one reading column under a centred title, with figures, tables and charts |
| `/benchmarks` | hybrid query time and ranking, the chart, every BEIR result (sortable), the mailbox, vector recall, build time and size, method |
| `/docs` | install, a first query, what's in it, the Python API, RAG, sparse fields, batches and snapshots, how it works, the Mac app |
| `/log` | how the benchmarks moved, commit by commit |

## The design system

One system for the library today and a hosted product later: a black or white page, greys, one
accent, pills for actions, hairlines instead of shadows, a lot of space.

- **Tokens** are in `src/tokens.stylex.ts`, defined with StyleX's `defineVars`: `colors`, `coats`
  (the mascot's colours), `fonts`, `type` (the type scale, which steps down on small screens),
  `space`, `radii`, `sizes`, and `media` (the breakpoints, with `defineConsts`). A component never
  writes a raw value for something a token names. Import the tokens by a relative path
  (`../tokens.stylex`), which is how StyleX finds them.
- **Shared styles** are in `src/components/styles.ts`: the type roles (`text.h1`, `text.lede`, …),
  the page frame (`frame.page`, `frame.wide`, `frame.prose`), links, the focus ring.
- **Components** are in `src/components`, each with its styles beside it:

  | | |
  |---|---|
  | `TopBar`, `Footer` | drawn from `src/data/nav.ts` |
  | `Button` | `button('primary' \| 'secondary' \| 'outline')` gives a link or a button the pill look |
  | `Link` | React Aria's link wrapped with TanStack Router's `createLink`: typed `to`, no page load |
  | `Page` | `PageHead` and `Section`, what every page but the home page is made of |
  | `Prose` | paragraphs, headings, lists, notes and callouts, for posts and docs |
  | `Figure`, `Table`, `CodeBlock`, `Segmented` | a captioned figure, a table from arrays, code with tabs and a copy button, a row of pills of which one is chosen |
  | `ChartParts`, `BarChart`, `LatencyStrip`, `Chart`, `LogChart` | the charts and what they share: the frame, tooltips, the entry animations |
  | `Mascot` | the hinny, in four coats |
  | `HomeParts` | the parts the home page and the announcement share: the kinds of search, the showcase, the panel about the name |
  | `Topology` | the engine cards |

- **Light and dark.** The page follows the reader's system; there is no switch. Colour tokens
  are `light-dark(light, dark)` values and the page's `color-scheme` is `light dark`. This needs
  Chrome 123, Safari 17.5 or Firefox 120.
- **Fonts** are Geist and Geist Mono (SIL Open Font License 1.1, `public/fonts/Geist-OFL.txt`),
  served from `public/fonts`: no third-party font requests.
- **Interactive parts** are React Aria's: the top bar's menu and the phone menu (a modal sheet
  with sections that fold), tabs on code, the pills over charts and tables, the sortable results
  table, tooltips on chart marks, the docs' "On this page" on small screens. They bring the
  keyboard and focus behaviour; StyleX styles them, through `className` functions where a style
  depends on state (`isSelected`).
- **Code** is written as plain text and coloured by a small tokenizer (`components/highlight.tsx`:
  comments, strings, numbers, keywords, calls; Python, Rust, C and a shell prompt). The colours
  are the `syntax` tokens.
- **The search showcase** on the home page (`src/stageshow/SearchShowcase.tsx`) is three faux
  applications with Hinny inside: a mailbox whose search finds by words and by meaning, an
  assistant whose tool runs a Hinny query, and a coding agent in a terminal that remembers with
  an upsert and recalls with a query. It is built on the stageshow kit, copied into
  `src/stageshow/kit` (MIT, `src/stageshow/LICENSE`). The kit is the one place with plain CSS
  instead of StyleX: its styles are in their own cascade layer and its colours are pointed at
  the site's tokens, so the stage follows light and dark. It needs `lucide-react`. The people,
  mails, notes and dates in it are made up.
- **The engine cards** (`components/Topology.tsx`) are small SVG diagrams of how a query runs
  and how a write is kept: a lead card across the top, then three to a row. They are on the
  announcement, with a line under them giving their figures' conditions. In the full site they
  are only on `/lab`, a page nothing links to: give them a place on the home page or leave them
  out, and delete `src/routes/lab.tsx` before the full site goes live.
- **Motion** is CSS, so the prerendered pages move without scripts: the home page's headline
  comes up word by word, cards rise as they are first seen, the hinny blinks and flicks an ear,
  and the one in the hero turns its eye towards the pointer. All of it stops under
  `prefers-reduced-motion`.
- **Charts** come in as they are first seen (bars grow, lines draw, marks pop in), slide when a
  tab changes what they show, and dim the other marks while one is pointed at. The entry is CSS
  held back by `useReveal` until the chart scrolls into view; with reduced motion they appear
  finished. Hinny's marks are the accent colour and a square; other systems are grey and round.

### How StyleX is wired

`@stylexjs/unplugin` in `vite.config.ts`, before the framework plugins. It compiles the
`stylex.create` calls to atomic CSS and appends it to the one stylesheet the app imports,
`src/styles.css`, which holds only what StyleX has no place for: the `@font-face` rules and a
reset. The reset is in a cascade layer (`reset`) declared before StyleX's layers, so component
styles always win. In development, `components/DevStyles.tsx` links the plugin's virtual
stylesheet, as StyleX's Vite guide describes for apps whose entry is a component.

## The announcement version

`VITE_ANNOUNCE=1 pnpm build` builds the site as one page that says Hinny is coming and shows
what it will be. It is about the engine only: the mascot, the headline and a plain line saying
"Coming soon", then the home page's own parts: the kinds of search (as plain cards, since there
are no docs to lead to), the engine cards with their figures' conditions, and the panel about
the name. The search showcase (Hinny inside applications) is left for the full site. There are
no badges or chips on it: its status is a sentence. There is no navigation, and the docs,
benchmarks, news and dev log are not built; their routes lead back to the home page. It
promises no date and takes no sign-ups. Its share image is `public/og-soon.png` and its
description starts "Coming soon". Without the variable the whole site is built. The switch is
`ANNOUNCE` in `src/data/site.ts`; the page is `src/components/Announcement.tsx`. On Vercel, set
`VITE_ANNOUNCE=1` in the project's environment variables for as long as only the announcement
should be live.

## What a shared link shows

A link to the site pasted into a chat or a post shows a card: a title, a description and an
image. The root route (`src/routes/__root.tsx`) sets the image (`public/og.png`, 1200 × 630;
`public/og-soon.png` for the announcement), its size and alternative text, the site's name and
the card's large format. Each page sets its own title, description and address with `pageMeta`
(`src/seo.ts`); a post does the same from its title and summary. The image's address is
absolute (`SITE_URL`), so a card only has its image once the site is served from that domain.

## Release state: `src/data/site.ts`

Everything the site says about availability comes from three constants in `src/data/site.ts`:

| Constant | Now | What it does |
|---|---|---|
| `RELEASED` | `false` | While false (the alpha), the home page, the docs and every post say "Alpha: builds from source", and the docs show how to build it from the repository. Set it to `true` once packages are published and they show `pip install hinny` and `cargo add hinny`. |
| `GITHUB_URL` | `https://github.com/janekwunderlich/hinny` | The link in the top bar, the footer and the docs. Empty hides every GitHub link. |
| `DOMAIN` | `hinny.dev` | Used for absolute URLs (the share image). |

## Writing a post

A post is one file in `src/news` that exports a `Post` (see `src/news/index.ts` for the fields):
its slug, title, date, category, summary, a cover, and a `Body` made of the elements in
`components/Prose.tsx`, `Figure`, `DataTable`, `CodeTabs`, the charts, and the small figures in
`src/news/parts.tsx`. Add it to the list in `src/news/index.ts` and it is on the news page, the
home page, the footer and at `/news/<slug>`. A figure with `wide` breaks out of the reading
column where the window has room. Categories are `Engineering`, `Product` and `Company`; an
announcement is a post like any other.

Every number in a post must come from the repository (`README.md`, `DESIGN.md`, `src/data`) and
keep its conditions: the machine, the data, the recall.

## Adding a product (a hosted Hinny, pricing)

Nothing on the site says there is a hosted product, and nothing should until there is one. When
there is, it is routes and data, not a redesign:

1. **A product page**: `src/routes/cloud.tsx`, built from `PageHead` and `Section`
   (`components/Page.tsx`), or from the home page's sections for a larger launch.
2. **A pricing page**: `src/routes/pricing.tsx`. Plans are cards on `colors.surface` like the
   home page's "Choose where to start", with `button('primary')` for the plan to pick; a
   comparison is a `DataTable` with `tintColumn`.
3. **Navigation**, in `src/data/nav.ts`. The top bar takes links and groups (a group is a menu):

   ```ts
   export const NAV: NavItem[] = [
     { label: 'Product', items: [
       { label: 'Hinny', to: '/', note: 'The embedded library' },
       { label: 'Hinny Cloud', to: '/cloud', note: '…' },
     ] },
     { label: 'Docs', items: [ … ] },
     { label: 'Benchmarks', to: '/benchmarks' },
     { label: 'Pricing', to: '/pricing' },
     { label: 'News', to: '/news' },
   ]
   ```

   and the footer's `Product` column gets the same two links. `to` is checked against the routes
   that exist, so a link to a page that isn't there doesn't compile.
4. **The pill in the top bar** is `Action` in `components/TopBar.tsx`: today "Read the docs"; a
   sign-in or a sign-up goes there, with `button('secondary')` beside it.
5. **The announcement** is a post with `category: 'Product'`.

## Benchmark numbers

The numbers come from `../benchmarks/beir/results.jsonl`: `pnpm data` (also the first step of
`pnpm build`) writes them to `src/data/beir.json`, which is committed, so a build with only this
folder still works. The rows carry the engine's ids as the benchmark harness wrote them, from
before the rename; `ENGINE` in `src/data/beir.ts` holds those ids and maps them to the labels the
site shows. Sentences that quote a range (`hybridSpeed`, `hybridTimes`) are computed from the
rows, so they can't drift from the charts beside them.

## The mark and the mascot

The mark is a hinny's head in profile, filled in, drawn for this site (its outline is `MARK` in
`Brand.tsx`; closed and filled, it is the mark). The same paths are in
`src/components/Brand.tsx` (the page), `public/hinny-mark.svg`, `public/favicon.svg` and
`scripts/og.html`; change all four together. The mascot (`src/components/Mascot.tsx`) is the
same drawing in four coats (iris, sun, clover and rose), and it moves. The
PNGs (`favicon-32.png`, `apple-touch-icon.png`, `og.png` and `og-soon.png`, the share images) are made from them:

```bash
sh scripts/brand-assets.sh   # needs rsvg-convert, magick and a Chromium
```

## Running it

```bash
pnpm install
pnpm dev           # http://localhost:3100
pnpm build         # prerendered pages and the server in .output (.vercel/output on Vercel)
pnpm typecheck
```

`pnpm build` (or `pnpm dev`) also regenerates `src/routeTree.gen.ts`; after adding a route, run
one of them before `pnpm typecheck`.

## Deploying on Vercel

Import the repository in Vercel and set **Root Directory** to `website`. Vercel detects the rest:
it runs `pnpm install` and `pnpm build` (the version in `packageManager`), and Nitro writes
`.vercel/output`, where the pages are static files. Nothing else needs configuring. To deploy
from a terminal instead:

```bash
cd website
npx vercel        # a preview deployment; --prod for production
```

To check the Vercel build locally: `NITRO_PRESET=vercel pnpm build`.

## Deploying on Cloudflare

The pages are static files, so Cloudflare needs no server code: `wrangler.jsonc` serves
`.output/public` as a Worker's static assets. From a terminal:

```bash
cd website
VITE_ANNOUNCE=1 pnpm build   # the announcement; plain `pnpm build` for the full site
npx wrangler login           # once: opens a browser to sign in to Cloudflare
npx wrangler deploy          # uploads .output/public; prints the workers.dev address
```

`npx wrangler deploy --dry-run` checks the configuration and the files without uploading. For
a domain of your own that is on the same Cloudflare account, add it under the Worker's
Settings, Domains & Routes, or in `wrangler.jsonc`:
`"routes": [{ "pattern": "hinny.dev", "custom_domain": true }]`.

Cloudflare builds and deploys the site on every push (Workers Builds: the `hinny-site` Worker,
Settings, Build, connected to this repository). Its settings:

| | |
|---|---|
| Root directory | `website` |
| Build command | `pnpm build` |
| Deploy command | `npx wrangler deploy` |
| Preview command | `npx wrangler versions upload` |
| Build variable `NITRO_PRESET` | `node-server`: without it Nitro builds for Cloudflare and the prerender waits forever on a local server |
| Build variable `VITE_ANNOUNCE` | `1` while only the announcement should be up; remove it (or `0`) for the whole site |

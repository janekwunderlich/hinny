import type { LinkProps } from '@tanstack/react-router'

/* The site's navigation, as data. The top bar, the phone menu and the footer are drawn from
   these two lists, so a new page (a product page, pricing) is added here and nowhere else.
   website/README.md, "Adding a product", shows what the lists look like with a second product. */

/** A link to a page of this site, with an optional anchor and a line of description. */
export type NavLink = { label: string; to: NonNullable<LinkProps['to']>; hash?: string; note?: string }

/** An entry of the top bar: a link, or a menu of links under one label. */
export type NavItem = NavLink | { label: string; items: NavLink[] }

/** The top bar, left to right. A group becomes a menu (a section of the phone menu). */
export const NAV: NavItem[] = [
  {
    label: 'Docs',
    items: [
      { label: 'Overview', to: '/docs', note: 'Core concepts, install, and architecture' },
      { label: 'A first query', to: '/docs', hash: 'first', note: 'Schema, upsert, and a filtered hybrid query' },
      { label: 'Python API', to: '/docs', hash: 'api', note: 'Namespaces, writes, queries, and filters' },
      { label: 'How it works', to: '/docs', hash: 'how', note: 'WAL, immutable segments, and 1-bit vectors' },
    ],
  },
  { label: 'Benchmarks', to: '/benchmarks' },
  { label: 'News', to: '/news' },
]

/** The footer's columns. */
export const FOOTER: { label: string; items: NavLink[] }[] = [
  {
    label: 'Product',
    items: [
      { label: 'Overview', to: '/' },
      { label: 'Benchmarks', to: '/benchmarks' },
    ],
  },
  {
    label: 'Developers',
    items: [
      { label: 'Docs', to: '/docs' },
      { label: 'Install', to: '/docs', hash: 'install' },
      { label: 'Python API', to: '/docs', hash: 'api' },
      { label: 'How it works', to: '/docs', hash: 'how' },
      { label: 'Dev log', to: '/log' },
    ],
  },
]

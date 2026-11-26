import type { ReactNode } from 'react'
import type { Coat } from '#/components/Mascot'
import { storage } from './every-file-can-be-checked'
import { vectors } from './a-million-vectors'

/* The news. A post is one file in this folder that exports a Post; listing it here publishes it:
   the news page, the home page's latest news, the footer and the post's own page at
   /news/<slug> all come from this list. website/README.md, "Writing a post", has the steps. */

export type Post = {
  /** The post's address: /news/<slug>. */
  slug: string
  title: string
  /** A shorter title, for the footer. */
  short: string
  /** The day it was published, as YYYY-MM-DD. */
  date: string
  /** What kind of news it is. "Product" and "Company" are for announcements. */
  category: 'Engineering' | 'Product' | 'Company'
  /** The hinny's coat on the post's cover. */
  coat: Coat
  /** One or two sentences, for lists and for the page's description. */
  summary: string
  /** The paragraph under the title on the post's own page. */
  lede: string
  /** A small drawing for the news page, where the latest post is shown large. */
  Cover: () => ReactNode
  /** The post itself: the elements of components/Prose.tsx, figures, tables and code. */
  Body: () => ReactNode
}

/** Every post, newest first. Posts of one day keep the order they are listed in here. */
export const posts: Post[] = [vectors, storage].sort((a, b) => b.date.localeCompare(a.date))

/** A post's date as it is shown: 13 Nov 2026. */
export const formatDate = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })

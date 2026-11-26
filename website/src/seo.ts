import { SITE_URL } from '#/data/site'

/** A page's head: its title and description, and the same again for the card a shared link
 * shows (the image and the rest come from the root route). `path` is the page's own, from the
 * root: '/docs'. */
export function pageMeta(title: string, description: string, path: string) {
  return [
    { title },
    { name: 'description', content: description },
    { property: 'og:title', content: title },
    { property: 'og:description', content: description },
    { property: 'og:url', content: `${SITE_URL}${path}` },
  ]
}

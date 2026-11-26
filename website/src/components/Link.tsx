import { createLink } from '@tanstack/react-router'
import { Link as AriaLink, MenuItem as AriaMenuItem } from 'react-aria-components'

/* Links are React Aria's (press handling, keyboard, focus) wrapped with TanStack Router's
   createLink, so they take a typed `to`, navigate without a page load and preload on intent.
   For an address outside the site, use ExternalLink. */

/** A link to a page of this site: `<Link to="/docs" hash="api">`. */
export const Link = createLink(AriaLink)

/** A menu item that is a link to a page of this site. */
export const MenuItemLink = createLink(AriaMenuItem)

/** A link to somewhere else: `<ExternalLink href="https://…">`. */
export const ExternalLink = AriaLink

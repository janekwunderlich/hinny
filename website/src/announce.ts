import { redirect } from '@tanstack/react-router'
import { ANNOUNCE } from '#/data/site'

/** For a route's `beforeLoad`: while the site is only the announcement, every page but the
 * home page leads back to it. */
export function announceOnly() {
  if (ANNOUNCE) throw redirect({ to: '/' })
}

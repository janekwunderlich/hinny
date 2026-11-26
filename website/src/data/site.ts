/** The few facts about the project's status that the whole site reads. Change them here, nowhere
 * else: every page, install snippet and link follows these. */

/** The product's name, as written in running text. */
export const NAME = 'Hinny'

/** The domain the site is served from. */
export const DOMAIN = 'hinny.dev'
export const SITE_URL = `https://${DOMAIN}`

/** False until the packages are published. While it is false, every place that would show an
 * install command says the alpha builds from source instead: the `hinny` names on crates.io,
 * PyPI and npm hold empty 0.0.1 placeholders. Flip it to true once real packages are up and the
 * commands below appear. */
export const RELEASED: boolean = false

/** The public source repository: linked from the header, footer and docs. Empty hides every
 * GitHub link. */
export const GITHUB_URL: string = 'https://github.com/janekwunderlich/hinny'

/** What to type once RELEASED is true. The Node.js binding (bindings/node) is not published to
 * npm, so it has no line here. */
export const INSTALL = [
  { name: 'Python', command: 'pip install hinny' },
  { name: 'Rust', command: 'cargo add hinny' },
] as const

/** The state every install slot shows while RELEASED is false. */
export const UNRELEASED = 'Alpha: builds from source'

/** True while the site is only an announcement that Hinny is coming: one page, no docs,
 * benchmarks or news, nothing to click through to. Set by building with VITE_ANNOUNCE=1
 * (`VITE_ANNOUNCE=1 pnpm build`); without it the whole site is built. */
export const ANNOUNCE: boolean = import.meta.env.VITE_ANNOUNCE === '1'

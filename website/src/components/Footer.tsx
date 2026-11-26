import * as stylex from '@stylexjs/stylex'
import { FOOTER } from '#/data/nav'
import { posts } from '#/news'
import { ANNOUNCE, GITHUB_URL, NAME } from '#/data/site'
import { colors, media, sizes, space, type } from '../tokens.stylex'
import { Logo } from './Brand'
import { ExternalLink, Link } from './Link'
import { focus, frame, link, text } from './styles'

/* The footer: the logo and a line saying what Hinny is at the left, columns of links at the right.
   The columns come from data/nav.ts, plus one for the latest news. */

const styles = stylex.create({
  foot: { marginTop: space.section, borderTopWidth: sizes.hair, borderTopStyle: 'solid', borderTopColor: colors.line },
  grid: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 1fr) minmax(0, 2fr)', [media.tablet]: 'minmax(0, 1fr)' },
    gap: space.x12,
    paddingTop: space.x10,
    paddingBottom: `max(${space.x16}, env(safe-area-inset-bottom, 0px))`,
  },
  about: { display: 'grid', gap: space.x4, alignContent: 'start', justifyItems: 'start' },
  brand: { display: 'inline-flex', textDecorationLine: 'none' },
  note: { maxWidth: '34ch' },
  columns: {
    display: 'grid',
    gridTemplateColumns: { default: 'repeat(3, minmax(0, 1fr))', [media.phone]: 'repeat(2, minmax(0, 1fr))' },
    gap: space.x8,
  },
  column: { display: 'grid', gap: space.x2, alignContent: 'start' },
  heading: { fontSize: type.meta, lineHeight: '20px', fontWeight: 500, color: colors.fg, marginBottom: space.x1 },
  item: { fontSize: type.meta, lineHeight: '21px' },
})

export function Footer() {
  return (
    <footer {...stylex.props(styles.foot)}>
      <div {...stylex.props(frame.page, styles.grid)}>
        <div {...stylex.props(styles.about)}>
          <Link to="/" aria-label={`${NAME}, home`} {...stylex.props(styles.brand, focus.ring)}>
            <Logo size={24} />
          </Link>
          <p {...stylex.props(text.meta, styles.note)}>Embedded hybrid search in Rust. Bindings for Python, Node.js, and C.</p>
          <p {...stylex.props(text.micro)}>
            {NAME} · MIT / Apache-2.0
            {GITHUB_URL && (
              <>
                {' · '}
                <ExternalLink href={GITHUB_URL} {...stylex.props(link.quiet, focus.ring)}>
                  GitHub
                </ExternalLink>
              </>
            )}
          </p>
        </div>
        {!ANNOUNCE && (
        <nav aria-label="Pages" {...stylex.props(styles.columns)}>
          {FOOTER.map((column) => (
            <div key={column.label} {...stylex.props(styles.column)}>
              <h2 {...stylex.props(styles.heading)}>{column.label}</h2>
              {column.items.map((l) => (
                <Link key={l.label} to={l.to} hash={l.hash} {...stylex.props(link.quiet, styles.item, focus.ring)}>
                  {l.label}
                </Link>
              ))}
              {column.label === 'Developers' && GITHUB_URL && (
                <ExternalLink href={GITHUB_URL} {...stylex.props(link.quiet, styles.item, focus.ring)}>
                  GitHub
                </ExternalLink>
              )}
            </div>
          ))}
          <div {...stylex.props(styles.column)}>
            <h2 {...stylex.props(styles.heading)}>News</h2>
            <Link to="/news" {...stylex.props(link.quiet, styles.item, focus.ring)}>
              All posts
            </Link>
            {posts.slice(0, 3).map((p) => (
              <Link key={p.slug} to="/news/$slug" params={{ slug: p.slug }} {...stylex.props(link.quiet, styles.item, focus.ring)}>
                {p.short}
              </Link>
            ))}
          </div>
        </nav>
        )}
      </div>
    </footer>
  )
}

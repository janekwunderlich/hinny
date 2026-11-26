import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'
import { colors, radii, sizes, space } from '../tokens.stylex'
import { ExternalLink } from './Link'
import { focus, link, text } from './styles'

/* The elements of running text, for posts and the docs: paragraphs, headings, lists, a note, a
   callout. A post's body is these and figures, one after another; the template sets the column
   and the space between them. */

const styles = stylex.create({
  /** The space between blocks of a body. Headings add more above themselves. */
  body: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: space.x5 },
  h2: { marginTop: space.x8, scrollMarginTop: '96px' },
  h3: { marginTop: space.x4, fontSize: '18px', lineHeight: '26px', letterSpacing: '-0.01em' },
  figure: { marginBlock: space.x4 },
  list: { display: 'grid', gap: space.x3, paddingLeft: 0, listStyleType: 'none' },
  item: { position: 'relative', paddingLeft: space.x6 },
  bullet: { position: 'absolute', left: 0, top: 0, color: colors.muted, userSelect: 'none' },
  callout: {
    display: 'grid',
    gap: space.x3,
    padding: { default: space.x8, '@media (max-width: 639px)': space.x5 },
    backgroundColor: colors.surface,
    borderRadius: radii.card,
  },
  rule: { height: sizes.hair, borderWidth: 0, backgroundColor: colors.line, margin: 0 },
  strong: { fontWeight: 600, color: colors.fg },
})

/** The blocks of a post or a doc section, spaced. */
export function Body({ children }: { children: ReactNode }) {
  return <div {...stylex.props(styles.body)}>{children}</div>
}

export function P({ children }: { children: ReactNode }) {
  return <p {...stylex.props(text.body)}>{children}</p>
}

/** A section heading inside a post. */
export function H2({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <h2 id={id} {...stylex.props(text.h3, styles.h2)}>
      {children}
    </h2>
  )
}

export function H3({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <h3 id={id} {...stylex.props(text.h3, styles.h3)}>
      {children}
    </h3>
  )
}

/** A link in running text to somewhere outside the site. Inside it, use Link with link.inline. */
export function A({ href, children }: { href: string; children: ReactNode }) {
  return (
    <ExternalLink href={href} {...stylex.props(link.inline, focus.ring)}>
      {children}
    </ExternalLink>
  )
}

export function Strong({ children }: { children: ReactNode }) {
  return <strong {...stylex.props(styles.strong)}>{children}</strong>
}

/** A list of points, each a sentence or two. */
export function List({ children }: { children: ReactNode }) {
  return <ul {...stylex.props(styles.list)}>{children}</ul>
}

export function Item({ children }: { children: ReactNode }) {
  return (
    <li {...stylex.props(text.body, styles.item)}>
      <span {...stylex.props(styles.bullet)} aria-hidden="true">
        –
      </span>
      {children}
    </li>
  )
}

/** Small print under a figure or a section: conditions, caveats. */
export function Note({ children }: { children: ReactNode }) {
  return <p {...stylex.props(text.small)}>{children}</p>
}

/** A block set apart on a surface: the state of the project, a warning. */
export function Callout({ children }: { children: ReactNode }) {
  return <div {...stylex.props(styles.callout)}>{children}</div>
}

/** Space around a figure inside a body. */
export function FigureSpace({ children }: { children: ReactNode }) {
  return <div {...stylex.props(styles.figure)}>{children}</div>
}

export function Rule() {
  return <hr {...stylex.props(styles.rule)} />
}

import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'
import { colors, radii, sizes, space } from '../tokens.stylex'
import { text } from './styles'

/* A figure: a chart, a table or a diagram, with the caption that says what was measured, on
   what and how. In a post it is as wide as the reading column; `wide` lets it break out to the
   width of the post's title row where the window has room. `boxed` draws a hairline frame, for
   charts and diagrams; tables go without. */

const styles = stylex.create({
  figure: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: space.x4, minWidth: 0, margin: 0 },
  // The reading column is 720px and the wide row 976px: 128px out on each side, once the window
  // holds the wide row and its gutters.
  wide: { marginInline: { default: 0, '@media (min-width: 1040px)': '-128px' } },
  box: {
    minWidth: 0,
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.line,
    borderRadius: radii.card,
    padding: { default: space.x6, '@media (max-width: 639px)': space.x4 },
  },
  caption: { maxWidth: '76ch' },
})

export function Figure({ children, caption, wide = false, boxed = false, label }: { children: ReactNode; caption?: ReactNode; wide?: boolean; boxed?: boolean; label?: string }) {
  return (
    <figure aria-label={label} {...stylex.props(styles.figure, wide && styles.wide)}>
      {boxed ? <div {...stylex.props(styles.box)}>{children}</div> : children}
      {caption && <figcaption {...stylex.props(text.small, styles.caption)}>{caption}</figcaption>}
    </figure>
  )
}

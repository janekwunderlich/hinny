import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'
import { colors, media, sizes, space } from '../tokens.stylex'
import { frame, text } from './styles'

/* The parts every page but the home page is made of: a head (a small label, the title, a
   paragraph) and sections under hairlines, each with its heading and text in a narrow column at
   the left and its content at the right. A product page or a pricing page is these too. */

const styles = stylex.create({
  head: { display: 'grid', gap: space.x5, justifyItems: 'start', paddingTop: space.section, paddingBottom: space.block },
  title: { maxWidth: '18em' },
  lede: { maxWidth: '62ch' },
  section: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 4fr) minmax(0, 8fr)', [media.tablet]: 'minmax(0, 1fr)' },
    columnGap: space.x12,
    rowGap: space.x6,
    paddingBlock: space.block,
    borderTopWidth: sizes.hair,
    borderTopStyle: 'solid',
    borderTopColor: colors.line,
    scrollMarginTop: '72px',
  },
  aside: { display: 'grid', gap: space.x3, alignContent: 'start', minWidth: 0 },
  sticky: { position: { default: 'sticky', [media.tablet]: 'static' }, top: '96px' },
  content: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: space.x6, alignContent: 'start', minWidth: 0 },
  lead: { maxWidth: '46ch' },
})

/** The top of a page: a small label, the title, a paragraph. */
export function PageHead({ label, title, children }: { label: string; title: ReactNode; children?: ReactNode }) {
  return (
    <header {...stylex.props(frame.page, styles.head)}>
      <p {...stylex.props(text.eyebrow)}>{label}</p>
      <h1 {...stylex.props(text.h1, styles.title)}>{title}</h1>
      {children && <p {...stylex.props(text.lede, styles.lede)}>{children}</p>}
    </header>
  )
}

/** A section under a hairline: heading and text at the left, content at the right; stacked
 * below a laptop's width. */
export function Section({ id, title, lead, children }: { id: string; title: ReactNode; lead?: ReactNode; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} {...stylex.props(frame.page)}>
      <div {...stylex.props(styles.section)}>
        <div {...stylex.props(styles.aside)}>
          <div {...stylex.props(styles.aside, styles.sticky)}>
            <h2 id={`${id}-h`} {...stylex.props(text.h3)}>
              {title}
            </h2>
            {lead && <p {...stylex.props(text.body, text.muted, styles.lead)}>{lead}</p>}
          </div>
        </div>
        <div {...stylex.props(styles.content)}>{children}</div>
      </div>
    </section>
  )
}

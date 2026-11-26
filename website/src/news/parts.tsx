import * as stylex from '@stylexjs/stylex'
import type { CSSProperties, ReactNode } from 'react'
import { ExternalLink } from '#/components/Link'
import { Hinny, type Coat } from '#/components/Mascot'
import { Callout, P, Strong } from '#/components/Prose'
import { reveal, stagger, useReveal } from '#/components/ChartParts'
import { focus, link, text } from '#/components/styles'
import { GITHUB_URL, NAME, RELEASED } from '#/data/site'
import { colors, fonts, media, radii, sizes, space, type } from '../tokens.stylex'

/* Small figures posts are illustrated with: a row of numbers that narrows step by step, a bar
   divided into parts, a file drawn as blocks, and the cover a post shows on the news page. */

const styles = stylex.create({
  // A row of numbers, each with a line under it, separated by hairlines. Where the figure is not
  // at its full width (below 1040px) the steps stack, each a row: the number, then its words.
  steps: {
    display: 'grid',
    gridTemplateColumns: { default: 'repeat(var(--steps), minmax(0, 1fr))', '@media (max-width: 1039px)': 'minmax(0, 1fr)' },
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.line,
    borderRadius: radii.card,
    overflow: 'hidden',
    // The hairlines between cells are the grid's background showing through 1px gaps.
    gap: sizes.hair,
    backgroundColor: colors.line,
  },
  step: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (max-width: 1039px)': '8.5rem minmax(0, 1fr)' },
    columnGap: space.x4,
    rowGap: space.x2,
    alignContent: 'start',
    alignItems: 'baseline',
    padding: space.x5,
    backgroundColor: colors.bg,
  },
  stepWords: { display: 'grid', gap: space.x1 },
  number: { fontSize: { default: '28px', [media.phone]: '22px' }, lineHeight: 1.1, fontWeight: 500, letterSpacing: '-0.03em', color: colors.fg },
  share: { fontSize: type.micro, lineHeight: '16px', color: colors.muted, fontVariantNumeric: 'tabular-nums' },

  // A bar divided into parts, with a key under it.
  bar: { display: 'flex', gap: '2px', height: '28px', borderRadius: '4px', overflow: 'hidden' },
  part: { minWidth: '2px' },
  tones: { display: 'grid', gap: space.x2, marginTop: space.x4 },
  tone: { display: 'grid', gridTemplateColumns: '12px minmax(0, 1fr) auto', alignItems: 'baseline', columnGap: space.x3, fontSize: type.small, lineHeight: '20px', color: colors.text },
  chip: { width: '10px', height: '10px', alignSelf: 'center' },
  amount: { color: colors.fg, fontWeight: 500, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' },
  focusTone: { backgroundColor: colors.accent },
  midTone: { backgroundColor: colors.other },
  lowTone: { backgroundColor: colors.lineStrong },

  // A file as a row of blocks.
  blocks: { display: 'flex', flexWrap: 'wrap', gap: '4px' },
  block: {
    display: 'grid',
    alignContent: 'center',
    gap: '2px',
    minHeight: '64px',
    paddingBlock: space.x2,
    paddingInline: space.x3,
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.lineStrong,
    borderRadius: '6px',
    flexGrow: 1,
    flexBasis: '96px',
  },
  blockMarked: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  blockFaint: { borderStyle: 'dashed', flexGrow: 0, flexBasis: '48px', justifyItems: 'center' },
  blockName: { fontSize: type.small, lineHeight: '20px', fontWeight: 500, color: colors.fg },
  blockNote: { fontFamily: fonts.mono, fontSize: '11px', lineHeight: '16px', color: colors.muted },

  // The cover: a few lines of the post's own facts, set in mono on a surface.
  cover: {
    display: 'grid',
    alignContent: 'center',
    gap: space.x3,
    minHeight: { default: '320px', [media.phone]: '220px' },
    padding: { default: space.x10, [media.phone]: space.x6 },
    backgroundColor: colors.surface,
    borderRadius: radii.card,
    fontFamily: fonts.mono,
    fontSize: { default: '15px', [media.phone]: '13px' },
    lineHeight: 1.5,
    fontVariantLigatures: 'none',
  },
  coverLine: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', columnGap: space.x4, alignItems: 'baseline', paddingBottom: space.x3, borderBottomWidth: sizes.hair, borderBottomStyle: 'solid', borderBottomColor: colors.line, color: colors.muted },
  coverMascot: { marginBottom: space.x3 },
  coverLast: { borderBottomWidth: 0, paddingBottom: 0 },
  coverValue: { color: colors.fg, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' },
})

/** Numbers that narrow from left to right: what a query starts with, and what it ends up reading. */
export function Steps({ steps }: { steps: { number: string; label: ReactNode; share?: string }[] }) {
  return (
    <ol {...stylex.props(styles.steps)} style={{ '--steps': steps.length } as CSSProperties}>
      {steps.map((s) => (
        <li key={s.number} {...stylex.props(styles.step)}>
          <span {...stylex.props(styles.number)}>{s.number}</span>
          <span {...stylex.props(styles.stepWords)}>
            <span {...stylex.props(text.small)}>{s.label}</span>
            {s.share && <span {...stylex.props(styles.share)}>{s.share}</span>}
          </span>
        </li>
      ))}
    </ol>
  )
}

const TONES = { focus: styles.focusTone, mid: styles.midTone, low: styles.lowTone } as const

/** One bar divided into parts by size, each named with its amount under the bar. The part the
 * text is about takes the accent; the others are greys, and every part is labelled in words. */
export function PartsBar({ parts, label }: { parts: { name: string; amount: string; size: number; tone: keyof typeof TONES }[]; label: string }) {
  const [ref, phase] = useReveal<HTMLDivElement>()
  return (
    <div ref={ref}>
      <div role="img" aria-label={label} {...stylex.props(styles.bar)}>
        {parts.map((p, i) => (
          <span key={p.name} {...stylex.props(styles.part, TONES[p.tone], reveal(phase, 'grow'))} style={{ flexGrow: p.size, flexBasis: 0, ...stagger(i, 180) }} />
        ))}
      </div>
      <ul {...stylex.props(styles.tones)}>
        {parts.map((p) => (
          <li key={p.name} {...stylex.props(styles.tone)}>
            <span aria-hidden="true" {...stylex.props(styles.chip, TONES[p.tone])} />
            <span>{p.name}</span>
            <span {...stylex.props(styles.amount)}>{p.amount}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** A file drawn as blocks in the order they are stored. Marked blocks are the checksums. */
export function Blocks({ blocks, label }: { blocks: { name: string; note?: string; kind?: 'marked' | 'faint' }[]; label: string }) {
  return (
    <div role="img" aria-label={label} {...stylex.props(styles.blocks)}>
      {blocks.map((b, i) => (
        <div key={i} {...stylex.props(styles.block, b.kind === 'marked' && styles.blockMarked, b.kind === 'faint' && styles.blockFaint)}>
          <span {...stylex.props(styles.blockName)}>{b.name}</span>
          {b.note && <span {...stylex.props(styles.blockNote)}>{b.note}</span>}
        </div>
      ))}
    </div>
  )
}

/** A post's cover on the news page: a few of its own facts as lines. Decoration; the post has
 * them in full. */
export function Cover({ lines, coat }: { lines: [string, string][]; coat: Coat }) {
  return (
    <div aria-hidden="true" {...stylex.props(styles.cover)}>
      <span {...stylex.props(styles.coverMascot)}>
        <Hinny size={64} coat={coat} />
      </span>
      {lines.map(([name, value], i) => (
        <div key={name} {...stylex.props(styles.coverLine, i === lines.length - 1 && styles.coverLast)}>
          <span>{name}</span>
          <span {...stylex.props(styles.coverValue)}>{value}</span>
        </div>
      ))}
    </div>
  )
}

/** The note every post about the engine opens with until packages are published. */
export function InDevelopment() {
  if (RELEASED) return null
  return (
    <Callout>
      <P>
        <Strong>{NAME} is an alpha.</Strong> Builds from source on{' '}
        {GITHUB_URL ? (
          <ExternalLink href={GITHUB_URL} {...stylex.props(link.inline, focus.ring)}>
            GitHub
          </ExternalLink>
        ) : (
          'GitHub'
        )}
        ; no registry packages yet. Every measurement below comes from the repo's benchmarks on the hardware noted beside it.
      </P>
    </Callout>
  )
}

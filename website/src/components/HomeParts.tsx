import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'
import { NAME } from '#/data/site'
import { SearchShowcase } from '../stageshow/SearchShowcase'
import { coats, colors, fonts, media, radii, space, type } from '../tokens.stylex'
import { Chevron } from './Button'
import { reveal, stagger, useReveal } from './ChartParts'
import { Link } from './Link'
import { Hinny, type Coat } from './Mascot'
import { focus, link, text } from './styles'

/* The parts of the home page that the announcement can show as well: the kinds of search, the
   search showcase and the panel about the name. Each is the inside of a section; the page that
   uses it gives it the frame and the air around it. */

/** Words at the left, something to look at at the right; one column on tablets and phones. */
export const split = stylex.create({
  row: { display: 'grid', gridTemplateColumns: { default: 'minmax(0, 5fr) minmax(0, 7fr)', [media.tablet]: 'minmax(0, 1fr)' }, alignItems: 'center', columnGap: space.x16, rowGap: space.x10 },
  words: { display: 'grid', gap: space.x5, justifyItems: 'start' },
})

const styles = stylex.create({
  // The kinds of search: cards on a surface, each a line of the API and what it does.
  kinds: { display: 'grid', gridTemplateColumns: { default: 'repeat(3, minmax(0, 1fr))', [media.tablet]: 'repeat(2, minmax(0, 1fr))', [media.phone]: 'minmax(0, 1fr)' }, gap: space.x3 },
  kind: { display: 'grid', gridTemplateRows: 'minmax(0, 1fr) auto', gap: space.x8, minHeight: '232px', padding: space.x6, borderRadius: radii.card, backgroundColor: colors.surface },
  kindLink: { backgroundColor: { default: colors.surface, ':hover': colors.raised }, textDecorationLine: 'none', transitionProperty: 'background-color', transitionDuration: '150ms' },
  kindWide: { gridColumn: { default: 'span 2', [media.phone]: 'auto' } },
  snippet: { fontFamily: fonts.mono, fontSize: type.code, lineHeight: 1.7, fontVariantLigatures: 'none', color: colors.muted, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' },
  kindTop: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', alignItems: 'start', columnGap: space.x4 },
  // On a phone the four hinnies of the hybrid card stand over its query, which needs the width.
  kindTopWide: { gridTemplateColumns: { default: 'minmax(0, 1fr) auto', [media.phone]: 'minmax(0, 1fr)' }, rowGap: space.x4 },
  herd: { display: 'flex' },
  herdWide: { order: { default: 0, [media.phone]: -1 } },
  herdNext: { marginLeft: '-10px' },
  kindFoot: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', alignItems: 'end', columnGap: space.x4, rowGap: space.x1 },
  kindWords: { display: 'grid', gap: space.x1 },
  // Three lines' room, so the names of a row of cards stand on one line.
  kindText: { minHeight: { default: '66px', [media.tablet]: '44px', [media.phone]: 'auto' } },
  hl: { color: colors.fg },

  // The name: the page's one saturated panel.
  band: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 1fr) auto', [media.tablet]: 'minmax(0, 1fr)' },
    alignItems: 'center',
    columnGap: space.x16,
    rowGap: space.x10,
    padding: { default: space.x16, [media.tablet]: space.x10, [media.phone]: space.x6 },
    borderRadius: radii.card,
    backgroundColor: coats.band,
    backgroundImage: `linear-gradient(135deg, ${coats.band}, ${coats.bandDeep})`,
    color: coats.onBand,
  },
  bandWords: { display: 'grid', gap: space.x5, maxWidth: '34em' },
  onBand: { color: coats.onBand },
  onBandMuted: { color: coats.onBandMuted },
  sum: { display: 'flex', alignItems: 'flex-end', gap: { default: space.x5, [media.phone]: space.x3 }, justifySelf: { default: 'end', [media.tablet]: 'start' } },
  term: { display: 'grid', justifyItems: 'center', gap: space.x2, fontSize: type.meta, lineHeight: '20px', color: coats.onBandMuted },
  sign: { alignSelf: 'center', paddingBottom: space.x6, fontSize: type.h3, lineHeight: 1, fontWeight: 400, color: coats.onBandMuted },
})

function Hl({ children }: { children: ReactNode }) {
  return <span {...stylex.props(styles.hl)}>{children}</span>
}

/** What Hinny searches by. Each card shows the piece of a query that asks for it. */
const KINDS: { name: string; text: string; snippet: ReactNode; hash: string; coats: Coat[]; wide?: boolean }[] = [
  {
    name: 'Full text',
    coats: ['sun'],
    text: 'Block-max MaxScore BM25 over bit-packed postings. Snowball stemming in 18 languages.',
    snippet: (
      <>
        docs.query({'\n  '}
        <Hl>text="apartment lease"</Hl>
        {'\n'})
      </>
    ),
    hash: 'parts',
  },
  {
    name: 'Vectors',
    coats: ['iris'],
    text: '1-bit RaBitQ codes over shared clusters. Half-precision reranking only when the error bound requires it.',
    snippet: (
      <>
        docs.query({'\n  '}
        <Hl>vector=embedding</Hl>
        {'\n'})
      </>
    ),
    hash: 'parts',
  },
  {
    name: 'Learned sparse',
    coats: ['clover'],
    text: 'SPLADE-style term weights scored by dot product on the same posting lists as BM25.',
    snippet: (
      <>
        docs.query({'\n  '}
        <Hl>sparse={'{'}"terms": weights{'}'}</Hl>
        {'\n'})
      </>
    ),
    hash: 'sparse',
  },
  {
    name: 'Filters',
    coats: ['rose'],
    text: 'Ranges, sets, prefixes, globs, tags, and boolean logic evaluated inside the search, not bolted on after.',
    snippet: (
      <>
        docs.query({'\n  '}
        <Hl>filter=attr("year") &gt;= 2025</Hl>
        {'\n'})
      </>
    ),
    hash: 'api',
  },
  {
    name: 'Hybrid',
    coats: ['sun', 'iris', 'clover', 'rose'],
    text: 'BM25, vectors, sparse terms, and filters in one call. One filter across every retriever, one fused ranking back.',
    snippet: (
      <>
        hits = docs.query({'\n  '}
        <Hl>text="apartment lease",</Hl>
        {'\n  '}
        <Hl>vector=embedding,</Hl>
        {'\n  '}
        <Hl>filter=attr("year") &gt;= 2025,</Hl>
        {'\n'}){'  '}# one ranking
      </>
    ),
    hash: 'why',
    wide: true,
  },
]

/** The kinds of search as cards. On the site each leads to its part of the docs; where there
 * are no docs to lead to (the announcement), pass `linked={false}` and they are plain cards. */
export function Kinds({ linked = true }: { linked?: boolean }) {
  const [ref, phase] = useReveal<HTMLDivElement>()
  return (
    <div ref={ref} {...stylex.props(styles.kinds)}>
      {KINDS.map((k, i) => {
        const inside = (
          <>
            <span {...stylex.props(styles.kindTop, k.wide && styles.kindTopWide)}>
              <pre {...stylex.props(styles.snippet)}>{k.snippet}</pre>
              <span {...stylex.props(styles.herd, k.wide && styles.herdWide)}>
                {k.coats.map((c, n) => (
                  <span key={c} {...stylex.props(n > 0 && styles.herdNext)}>
                    <Hinny size={36} coat={c} delay={n * 1.3} />
                  </span>
                ))}
              </span>
            </span>
            <span {...stylex.props(styles.kindFoot)}>
              <span {...stylex.props(styles.kindWords)}>
                <span {...stylex.props(text.title)}>{k.name}</span>
                <span {...stylex.props(text.small, styles.kindText)}>{k.text}</span>
              </span>
              {linked && (
                <span {...stylex.props(link.more)}>
                  Docs
                  <Chevron />
                </span>
              )}
            </span>
          </>
        )
        return linked ? (
          <Link key={k.name} to="/docs" hash={k.hash} {...stylex.props(styles.kind, styles.kindLink, k.wide && styles.kindWide, focus.ringOnly, reveal(phase, 'rise'))} style={stagger(i, 80, 700)}>
            {inside}
          </Link>
        ) : (
          <div key={k.name} {...stylex.props(styles.kind, k.wide && styles.kindWide, reveal(phase, 'rise'))} style={stagger(i, 80, 700)}>
            {inside}
          </div>
        )
      })}
    </div>
  )
}

/** The search showcase beside what it shows. `headingId` names the heading for the section. */
export function Showcase({ headingId }: { headingId: string }) {
  return (
    <div {...stylex.props(split.row)}>
      <div {...stylex.props(split.words)}>
        <p {...stylex.props(text.eyebrow)}>Hybrid retrieval</p>
        <h2 id={headingId} {...stylex.props(text.h1)}>
          Keywords plus meaning.
          <br />
          In one query.
        </h2>
        <p {...stylex.props(text.lede)}>BM25 hits exact names, IDs, and error codes. Vectors catch paraphrases when the words don't overlap. Hinny runs both under the same filter and blends their scores.</p>
        <p {...stylex.props(text.small)}>Sample data below. In Mail, tags show which retriever matched each hit. In Assistant and Memory, the app's tools call Hinny directly.</p>
      </div>
      <SearchShowcase />
    </div>
  )
}

/** The panel about the name: a hinny is a hybrid, and so is a query. */
export function NameBand({ headingId }: { headingId: string }) {
  return (
    <div {...stylex.props(styles.band)}>
      <div {...stylex.props(styles.bandWords)}>
        <p {...stylex.props(text.eyebrow, styles.onBandMuted)}>Why &ldquo;Hinny&rdquo;</p>
        <h2 id={headingId} {...stylex.props(text.h1, styles.onBand)}>
          A hinny is a hybrid.
          <br />
          So is every query.
        </h2>
        <p {...stylex.props(text.lede, styles.onBandMuted)}>
          A hinny is the cross of a horse and a donkey. {NAME} crosses lexical search with vector search in a single local engine, so you don't have to glue two databases together.
        </p>
      </div>
      <div {...stylex.props(styles.sum)} role="img" aria-label="Words plus meaning make one ranking">
        <span {...stylex.props(styles.term)}>
          <Hinny size={72} coat="sun" delay={0.8} />
          words
        </span>
        <span {...stylex.props(styles.sign)}>+</span>
        <span {...stylex.props(styles.term)}>
          <Hinny size={72} coat="rose" delay={2.1} />
          meaning
        </span>
        <span {...stylex.props(styles.sign)}>=</span>
        <span {...stylex.props(styles.term)}>
          <Hinny size={112} coat="paper" delay={3.4} />
          one ranking
        </span>
      </div>
    </div>
  )
}

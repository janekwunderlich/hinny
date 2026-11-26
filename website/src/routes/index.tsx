import * as stylex from '@stylexjs/stylex'
import { createFileRoute } from '@tanstack/react-router'
import { Fragment } from 'react'
import { Announcement } from '#/components/Announcement'
import { Architecture } from '#/components/Architecture'
import { Chevron, button } from '#/components/Button'
import { motion, stagger } from '#/components/ChartParts'
import { CodeTabs, InlineCode } from '#/components/CodeBlock'
import { Kinds, NameBand, Showcase, split } from '#/components/HomeParts'
import { PrimaryAction } from '#/components/Install'
import { LatencyStrip } from '#/components/LatencyStrip'
import { Link } from '#/components/Link'
import { Hinny } from '#/components/Mascot'
import { NewsList } from '#/components/NewsList'
import { a11y, focus, frame, link, text } from '#/components/styles'
import { hybridSpeed, hybridTimes, ms } from '#/data/beir'
import { fourWays } from '#/data/samples'
import { ANNOUNCE, NAME, RELEASED, SITE_URL, UNRELEASED } from '#/data/site'
import { posts } from '#/news'
import { colors, media, radii, sizes, space, type } from '../tokens.stylex'

export const Route = createFileRoute('/')({
  head: () => ({ meta: [{ property: 'og:url', content: `${SITE_URL}/` }] }),
  component: ANNOUNCE ? Announcement : Home,
})

/** The headline's words tip up into place, one after another. */
const wordIn = stylex.keyframes({ from: { opacity: 0, transform: 'translateY(45%) rotateX(-40deg)' }, to: { opacity: 1, transform: 'translateY(0) rotateX(0deg)' } })
const HEADLINE = 'Hybrid search inside your process.'

const styles = stylex.create({
  // The hero: one large line, a paragraph, the actions; centred.
  hero: { display: 'grid', justifyItems: 'center', gap: space.x6, textAlign: 'center', paddingTop: space.section, paddingBottom: space.section },
  heroTitle: { maxWidth: '13em', perspective: '1200px' },
  word: { display: 'inline-block', transformOrigin: '50% 100%', animationName: wordIn, animationDuration: '800ms', animationTimingFunction: 'cubic-bezier(0.2, 0.7, 0.2, 1)', animationFillMode: 'both' },
  heroLede: { maxWidth: '40em', textWrap: 'balance' },
  announce: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: space.x2,
    maxWidth: '100%',
    minHeight: '36px',
    paddingBlock: '6px',
    paddingInlineStart: '6px',
    paddingInlineEnd: space.x3,
    borderRadius: radii.pill,
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.line,
    backgroundColor: { default: colors.surface, ':hover': colors.raised },
    fontSize: type.meta,
    lineHeight: '20px',
    color: colors.fg,
    textDecorationLine: 'none',
    transitionProperty: 'background-color',
    transitionDuration: '150ms',
  },
  tag: { paddingBlock: '2px', paddingInline: space.x2, borderRadius: radii.pill, backgroundColor: colors.raised, fontSize: type.micro, lineHeight: '16px', fontWeight: 500, color: colors.fg },
  announceTitle: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },

  demo: { paddingTop: space.section },
  arch: { paddingBottom: space.section },
  archWords: { display: 'grid', gap: space.x3 },
  waysList: { display: 'grid', gap: space.x4, marginBlock: space.x2 },
  way: { display: 'grid', gridTemplateColumns: '4.5rem minmax(0, 1fr)', columnGap: space.x4, alignItems: 'baseline' },

  // The numbers: three in a row between hairlines.
  numbers: { display: 'grid', gridTemplateColumns: { default: 'repeat(3, minmax(0, 1fr))', [media.tablet]: 'minmax(0, 1fr)' }, borderBlockWidth: sizes.hair, borderBlockStyle: 'solid', borderBlockColor: colors.line },
  number: {
    display: 'grid',
    gap: space.x3,
    alignContent: 'start',
    paddingBlock: space.x10,
    paddingInlineEnd: space.x8,
    paddingInlineStart: { default: space.x8, [media.tablet]: 0 },
    borderInlineStartWidth: { default: sizes.hair, [media.tablet]: 0 },
    borderInlineStartStyle: 'solid',
    borderInlineStartColor: colors.line,
    borderTopWidth: { default: 0, [media.tablet]: sizes.hair },
    borderTopStyle: 'solid',
    borderTopColor: colors.line,
  },
  numberFirst: { paddingInlineStart: 0, borderInlineStartWidth: 0, borderTopWidth: 0 },
  figure: { fontSize: type.h1, lineHeight: 1, fontWeight: 500, letterSpacing: '-0.035em', color: colors.fg },
  numberNote: { maxWidth: '34ch' },
  strip: { paddingTop: space.x8 },
  numbersFoot: { display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'baseline', columnGap: space.x8, rowGap: space.x3, paddingTop: space.x5 },
  conditions: { maxWidth: '76ch' },

  name: { paddingTop: space.section },

  // Section headings with a link at the right.
  heading: { display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', justifyContent: 'space-between', columnGap: space.x6, rowGap: space.x2, marginBottom: space.x8 },

  // Where to start: two cards.
  start: { display: 'grid', gap: space.x10 },
  startCards: { display: 'grid', gridTemplateColumns: { default: 'repeat(2, minmax(0, 1fr))', [media.tablet]: 'minmax(0, 1fr)' }, gap: space.x3 },
  card: { display: 'grid', gridTemplateRows: 'auto auto minmax(0, 1fr) auto', gap: space.x4, padding: { default: space.x10, [media.phone]: space.x6 }, borderRadius: radii.card, backgroundColor: colors.surface },
  checks: { display: 'grid', gap: space.x3, paddingBlock: space.x5, borderTopWidth: sizes.hair, borderTopStyle: 'solid', borderTopColor: colors.line, marginTop: space.x2 },
  check: { display: 'grid', gridTemplateColumns: '16px minmax(0, 1fr)', columnGap: space.x3, alignItems: 'start', fontSize: type.small, lineHeight: '22px', color: colors.text },
  tick: { marginTop: '4px', color: colors.muted },
})

function Tick() {
  return (
    <svg {...stylex.props(styles.tick)} width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="m3 7.5 2.5 2.5L11 4.5" />
    </svg>
  )
}

/** The home page: what Hinny is in one line, the kinds of search it does, the three ways to use
 * it, a few measured numbers with their conditions, the latest news, and where to start. */
function Home() {
  const latest = posts[0]
  return (
    <>
      <section {...stylex.props(frame.page, styles.hero)}>
        <span {...stylex.props(motion.pop)}>
          <Hinny size={88} coat="iris" follow />
        </span>
        {latest && (
          <Link to="/news/$slug" params={{ slug: latest.slug }} {...stylex.props(styles.announce, focus.ringOnly)}>
            <span {...stylex.props(styles.tag)}>News</span>
            <span {...stylex.props(styles.announceTitle)}>{latest.title}</span>
            <Chevron />
          </Link>
        )}
        <h1 {...stylex.props(text.display, styles.heroTitle)}>
          {HEADLINE.split(' ').map((word, i) => (
            <Fragment key={i}>
              <span {...stylex.props(styles.word)} style={stagger(i, 70, 120)}>
                {word}
              </span>{' '}
            </Fragment>
          ))}
        </h1>
        <p {...stylex.props(text.lede, styles.heroLede, motion.rise)} style={stagger(0, 0, 620)}>
          BM25 full text, dense vectors, learned sparse terms, and attribute filters in one embedded engine. Memory-mapped local files, instant writes, zero network hops. Built in Rust for Rust, Python, Node.js, and C.
        </p>
        <PrimaryAction />
      </section>

      <section aria-labelledby="kinds-h" {...stylex.props(frame.page)}>
        <h2 id="kinds-h" {...stylex.props(a11y.hidden)}>
          The kinds of search
        </h2>
        <Kinds />
      </section>

      <section aria-labelledby="demo-h" {...stylex.props(frame.page, styles.demo)}>
        <Showcase headingId="demo-h" />
      </section>

      <section aria-labelledby="ways-h" {...stylex.props(frame.page, frame.section)}>
        <div {...stylex.props(split.row)}>
          <div {...stylex.props(split.words)}>
            <p {...stylex.props(text.eyebrow)}>Embedded library</p>
            <h2 id="ways-h" {...stylex.props(text.h1)}>
              Call it like
              <br />
              a function.
            </h2>
            <p {...stylex.props(text.lede)}>Same on-disk format, same query planner, same ranking from every language. No sidecar, no daemon, no network round trip.</p>
            <dl {...stylex.props(styles.waysList)}>
              <div {...stylex.props(styles.way)}>
                <dt {...stylex.props(text.title)}>Rust</dt>
                <dd {...stylex.props(text.small)}>
                  Core engine via the <InlineCode>hinny</InlineCode> crate.
                </dd>
              </div>
              <div {...stylex.props(styles.way)}>
                <dt {...stylex.props(text.title)}>Python</dt>
                <dd {...stylex.props(text.small)}>PyO3 extension. Reads NumPy float32 arrays in place and releases the GIL during queries.</dd>
              </div>
              <div {...stylex.props(styles.way)}>
                <dt {...stylex.props(text.title)}>Node.js</dt>
                <dd {...stylex.props(text.small)}>Native napi-rs addon with TypeScript types and zero runtime dependencies.</dd>
              </div>
              <div {...stylex.props(styles.way)}>
                <dt {...stylex.props(text.title)}>C</dt>
                <dd {...stylex.props(text.small)}>
                  One header, <InlineCode>hinny.h</InlineCode>: JSON payloads and raw float arrays for Swift, Kotlin, or C FFI.
                </dd>
              </div>
            </dl>
            <Link to="/docs" hash="first" {...button('secondary')}>
              A first query
              <Chevron />
            </Link>
          </div>
          <CodeTabs tabs={fourWays} note={RELEASED ? undefined : 'API preview'} />
        </div>
      </section>

      <section aria-labelledby="arch-h" {...stylex.props(frame.page, styles.arch)}>
        <div {...stylex.props(styles.heading)}>
          <div {...stylex.props(styles.archWords)}>
            <p {...stylex.props(text.eyebrow)}>Architecture</p>
            <h2 id="arch-h" {...stylex.props(text.h2)}>
              WAL, immutable segments, lock-free reads.
            </h2>
          </div>
          <Link to="/docs" hash="how" {...stylex.props(link.more, focus.ring)}>
            How it works
            <Chevron />
          </Link>
        </div>
        <Architecture />
      </section>

      <section aria-labelledby="numbers-h" {...stylex.props(frame.page)}>
        <h2 id="numbers-h" {...stylex.props(a11y.hidden)}>
          Measured
        </h2>
        <dl {...stylex.props(styles.numbers)}>
          <div {...stylex.props(styles.number, styles.numberFirst)}>
            <dt {...stylex.props(styles.figure)}>{ms(hybridTimes.fiqa)}</dt>
            <dd {...stylex.props(text.small, styles.numberNote)}>p50 hybrid query (BM25 + vectors, top 10) over 57,638 FiQA docs from Python</dd>
          </div>
          <div {...stylex.props(styles.number)}>
            <dt {...stylex.props(styles.figure)}>0.13 ms</dt>
            <dd {...stylex.props(text.small, styles.numberNote)}>p50 vector query over 1M 256-dim clustered vectors at 0.997 recall@10</dd>
          </div>
          <div {...stylex.props(styles.number)}>
            <dt {...stylex.props(styles.figure)}>
              {hybridSpeed.lanceLow}–{hybridSpeed.lanceHigh}×
            </dt>
            <dd {...stylex.props(text.small, styles.numberNote)}>faster than LanceDB across five BEIR hybrid benchmarks, with higher nDCG@10 on all five</dd>
          </div>
        </dl>
        <div {...stylex.props(styles.strip)}>
          <LatencyStrip />
        </div>
        <div {...stylex.props(styles.numbersFoot)}>
          <p {...stylex.props(text.small, styles.conditions)}>Single-threaded, warm cache on Apple M5 Pro. Every BEIR system gets identical documents, embeddings, and queries. Reproducible scripts live in the repo.</p>
          <Link to="/benchmarks" {...stylex.props(link.more, focus.ring)}>
            Full benchmarks and method
            <Chevron />
          </Link>
        </div>
      </section>

      <section aria-labelledby="name-h" {...stylex.props(frame.page, styles.name)}>
        <NameBand headingId="name-h" />
      </section>

      <section aria-labelledby="news-h" {...stylex.props(frame.page, frame.section)}>
        <div {...stylex.props(styles.heading)}>
          <h2 id="news-h" {...stylex.props(text.h2)}>
            Latest news
          </h2>
          <Link to="/news" {...stylex.props(link.more, focus.ring)}>
            All posts
            <Chevron />
          </Link>
        </div>
        <NewsList posts={posts.slice(0, 4)} />
      </section>

      <section aria-labelledby="start-h" {...stylex.props(frame.page, frame.ruled, frame.section, styles.start)}>
        <h2 id="start-h" {...stylex.props(text.h2, text.center)}>
          Get started
        </h2>
        <div {...stylex.props(styles.startCards)}>
          <div {...stylex.props(styles.card)}>
            <h3 {...stylex.props(text.h3)}>Build from source</h3>
            <p {...stylex.props(text.body)}>Clone the repo, run a query in five lines, and read how the storage engine works.</p>
            <ul {...stylex.props(styles.checks)}>
              {['In-process on memory-mapped local files', 'Rust, Python, Node.js, C, plus optional HTTP server', 'Dual-licensed MIT / Apache-2.0', UNRELEASED].map((c) => (
                <li key={c} {...stylex.props(styles.check)}>
                  <Tick />
                  {c}
                </li>
              ))}
            </ul>
            <Link to="/docs" {...button('primary', 'md', true)}>
              Read the docs
            </Link>
          </div>
          <div {...stylex.props(styles.card)}>
            <h3 {...stylex.props(text.h3)}>Inspect the benchmarks</h3>
            <p {...stylex.props(text.body)}>Raw numbers, reproducible scripts, and every caveat we measured.</p>
            <ul {...stylex.props(styles.checks)}>
              {['5 BEIR datasets, 6 embedded databases', 'Quality and latency from the exact same queries', '1M-vector comparison against FAISS, usearch, hnswlib', 'Commit-by-commit dev log on every change'].map((c) => (
                <li key={c} {...stylex.props(styles.check)}>
                  <Tick />
                  {c}
                </li>
              ))}
            </ul>
            <Link to="/benchmarks" {...button('outline', 'md', true)}>
              See the benchmarks
            </Link>
          </div>
        </div>
      </section>
    </>
  )
}

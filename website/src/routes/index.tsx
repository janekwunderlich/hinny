import * as stylex from '@stylexjs/stylex'
import { createFileRoute } from '@tanstack/react-router'
import { Fragment } from 'react'
import { Announcement } from '#/components/Announcement'
import { Chevron, button } from '#/components/Button'
import { motion, stagger } from '#/components/ChartParts'
import { Kinds, NameBand } from '#/components/HomeParts'
import { PrimaryAction } from '#/components/Install'
import { Link } from '#/components/Link'
import { Hinny } from '#/components/Mascot'
import { EngineCards } from '#/components/Topology'
import { a11y, focus, frame, link, text } from '#/components/styles'
import { ANNOUNCE, SITE_URL, UNRELEASED } from '#/data/site'
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

  engine: { paddingTop: space.section },
  engineWords: { display: 'grid', gap: space.x4, maxWidth: '44em' },
  engineFoot: { display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'baseline', columnGap: space.x8, rowGap: space.x3, paddingTop: space.x8 },
  conditions: { maxWidth: '86ch' },

  name: { paddingTop: space.section },

  // Section headings with a link at the right.
  heading: { display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', justifyContent: 'space-between', columnGap: space.x6, rowGap: space.x2, marginBottom: space.x10 },

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

/** The home page, laid out like the announcement it replaced: the hero, the kinds of search,
 * inside the engine, the name, and where to start. */
function Home() {
  return (
    <>
      <section {...stylex.props(frame.page, styles.hero)}>
        <span {...stylex.props(motion.pop)}>
          <Hinny size={96} coat="iris" follow />
        </span>
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

      <section aria-labelledby="engine-h" {...stylex.props(frame.page, styles.engine)}>
        <div {...stylex.props(styles.heading)}>
          <div {...stylex.props(styles.engineWords)}>
            <p {...stylex.props(text.eyebrow)}>Architecture</p>
            <h2 id="engine-h" {...stylex.props(text.h1)}>
              Inside the engine.
            </h2>
            <p {...stylex.props(text.lede)}>How a query runs, how a write is kept, and what each choice measured.</p>
          </div>
          <Link to="/docs" hash="how" {...stylex.props(link.more, focus.ring)}>
            How it works
            <Chevron />
          </Link>
        </div>
        <EngineCards />
        <div {...stylex.props(styles.engineFoot)}>
          <p {...stylex.props(text.small, styles.conditions)}>
            The figures are from the engine's design notes and its tests. The million vectors are synthetic (256 dimensions, clustered), searched at recall@10 0.997. The 80,000 writes are documents with 384-dimension vectors, written one at a time while another thread queries without pause. Alpha results, measured on the code in the repository.
          </p>
          <Link to="/benchmarks" {...stylex.props(link.more, focus.ring)}>
            Full benchmarks and method
            <Chevron />
          </Link>
        </div>
      </section>

      <section aria-labelledby="name-h" {...stylex.props(frame.page, styles.name)}>
        <NameBand headingId="name-h" />
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

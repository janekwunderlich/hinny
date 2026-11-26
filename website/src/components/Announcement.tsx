import * as stylex from '@stylexjs/stylex'
import { Fragment } from 'react'
import { GITHUB_URL, UNRELEASED } from '#/data/site'
import { space } from '../tokens.stylex'
import { button, Chevron } from './Button'
import { motion, stagger } from './ChartParts'
import { Kinds, NameBand } from './HomeParts'
import { ExternalLink } from './Link'
import { Hinny } from './Mascot'
import { EngineCards } from './Topology'
import { a11y, frame, text } from './styles'

/* The site as an announcement: Hinny is coming, and what it will be. One page about the engine,
   made of the home page's own parts: what it searches by, how it is built, and the name. The
   applications it can be put in (the search showcase) are left for the full site. That it is
   coming is said in a plain line under the hero's text: no badge, no chips. This is the whole
   site while ANNOUNCE is on (data/site.ts). It promises no date and takes no sign-ups, because
   there is neither, and nothing on it leads to a page that isn't built. */

const wordIn = stylex.keyframes({ from: { opacity: 0, transform: 'translateY(45%) rotateX(-40deg)' }, to: { opacity: 1, transform: 'translateY(0) rotateX(0deg)' } })
const HEADLINE = 'Hybrid search inside your process.'

const styles = stylex.create({
  hero: { display: 'grid', justifyItems: 'center', gap: space.x6, textAlign: 'center', paddingTop: space.section, paddingBottom: space.block },
  title: { maxWidth: '13em', perspective: '1200px' },
  word: { display: 'inline-block', transformOrigin: '50% 100%', animationName: wordIn, animationDuration: '800ms', animationTimingFunction: 'cubic-bezier(0.2, 0.7, 0.2, 1)', animationFillMode: 'both' },
  lede: { maxWidth: '40em', textWrap: 'balance' },

  engine: { display: 'grid', gap: space.x10, paddingTop: space.section },
  engineWords: { display: 'grid', gap: space.x4, maxWidth: '44em' },
  conditions: { maxWidth: '86ch' },
  name: { paddingTop: space.section },
})

export function Announcement() {
  return (
    <>
      <section {...stylex.props(frame.page, styles.hero)}>
        <span {...stylex.props(motion.pop)}>
          <Hinny size={96} coat="iris" follow />
        </span>
        <h1 {...stylex.props(text.display, styles.title)}>
          {HEADLINE.split(' ').map((word, i) => (
            <Fragment key={i}>
              <span {...stylex.props(styles.word)} style={stagger(i, 70, 120)}>
                {word}
              </span>{' '}
            </Fragment>
          ))}
        </h1>
        <p {...stylex.props(text.lede, styles.lede, motion.rise)} style={stagger(0, 0, 620)}>
          BM25 full text, dense vectors, learned sparse terms, and attribute filters in one embedded engine. Memory-mapped local files, instant writes, zero network hops. Built in Rust for Rust, Python, Node.js, and C.
        </p>
        {GITHUB_URL && (
          <ExternalLink href={GITHUB_URL} {...button('primary')}>
            View on GitHub
            <Chevron />
          </ExternalLink>
        )}
        <p {...stylex.props(text.small, motion.rise)} style={stagger(0, 0, 760)}>
          {UNRELEASED}.
        </p>
      </section>

      <section aria-labelledby="kinds-h" {...stylex.props(frame.page)}>
        <h2 id="kinds-h" {...stylex.props(a11y.hidden)}>
          The kinds of search
        </h2>
        <Kinds linked={false} />
      </section>

      <section aria-labelledby="engine-h" {...stylex.props(frame.page, styles.engine)}>
        <div {...stylex.props(styles.engineWords)}>
          <p {...stylex.props(text.eyebrow)}>Architecture</p>
          <h2 id="engine-h" {...stylex.props(text.h1)}>
            Inside the engine.
          </h2>
          <p {...stylex.props(text.lede)}>How a query runs, how a write is kept, and what each choice measured.</p>
        </div>
        <EngineCards />
        <p {...stylex.props(text.small, styles.conditions)}>
          The figures are from the engine's design notes and its tests. The million vectors are synthetic (256 dimensions, clustered), searched at recall@10 0.997. The 80,000 writes are documents with 384-dimension vectors, written one at a time while another thread queries without pause. Development results: {UNRELEASED.toLowerCase()}.
        </p>
      </section>

      <section aria-labelledby="name-h" {...stylex.props(frame.page, styles.name)}>
        <NameBand headingId="name-h" />
      </section>
    </>
  )
}

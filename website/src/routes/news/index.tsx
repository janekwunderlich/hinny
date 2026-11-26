import * as stylex from '@stylexjs/stylex'
import { announceOnly } from '#/announce'
import { createFileRoute } from '@tanstack/react-router'
import { pageMeta } from '#/seo'
import { Chevron, button } from '#/components/Button'
import { Link } from '#/components/Link'
import { NewsList } from '#/components/NewsList'
import { a11y, focus, frame, link, text } from '#/components/styles'
import { NAME } from '#/data/site'
import { formatDate, posts } from '#/news'
import { colors, media, space } from '../../tokens.stylex'

export const Route = createFileRoute('/news/')({
  beforeLoad: announceOnly,
  head: () => ({ meta: pageMeta(`News · ${NAME}`, `News from ${NAME}: how the engine works, what was measured, and what changed.`, '/news') }),
  component: News,
})

const styles = stylex.create({
  featured: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 5fr) minmax(0, 7fr)', [media.tablet]: 'minmax(0, 1fr)' },
    alignItems: 'center',
    columnGap: space.x16,
    rowGap: space.x8,
    paddingTop: space.section,
    paddingBottom: space.block,
  },
  words: { display: 'grid', gap: space.x5, justifyItems: 'start' },
  meta: { display: 'flex', gap: space.x3 },
  category: { color: colors.fg, fontWeight: 500 },
  title: { textDecorationLine: 'none', color: 'inherit' },
  all: { display: 'grid', gap: space.x8, paddingTop: space.block },
  more: { display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', columnGap: space.x3, rowGap: space.x1, paddingTop: space.x2 },
})

/** The news page: the latest post large, with its cover, then every post as a row. */
function News() {
  const [latest] = posts
  return (
    <div {...stylex.props(frame.page)}>
      <h1 {...stylex.props(a11y.hidden)}>News</h1>
      {latest && (
        <section aria-label="The latest post" {...stylex.props(styles.featured)}>
          <div {...stylex.props(styles.words)}>
            <p {...stylex.props(text.meta, styles.meta)}>
              <span {...stylex.props(styles.category)}>{latest.category}</span>
              <time dateTime={latest.date}>{formatDate(latest.date)}</time>
            </p>
            <h2 {...stylex.props(text.h1)}>
              <Link to="/news/$slug" params={{ slug: latest.slug }} {...stylex.props(styles.title, focus.ring)}>
                {latest.title}
              </Link>
            </h2>
            <p {...stylex.props(text.lede)}>{latest.summary}</p>
            <Link to="/news/$slug" params={{ slug: latest.slug }} aria-label={`Read “${latest.title}”`} {...button('primary')}>
              Read the post
              <Chevron />
            </Link>
          </div>
          <latest.Cover />
        </section>
      )}
      <section aria-labelledby="all-h" {...stylex.props(styles.all)}>
        <h2 id="all-h" {...stylex.props(text.h2)}>
          All posts
        </h2>
        <NewsList posts={posts} />
        <p {...stylex.props(text.small, styles.more)}>
          Every engine change and its benchmark impact, commit by commit, is tracked in the
          <Link to="/log" {...stylex.props(link.more, focus.ring)}>
            dev log
            <Chevron />
          </Link>
        </p>
      </section>
    </div>
  )
}

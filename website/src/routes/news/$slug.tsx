import * as stylex from '@stylexjs/stylex'
import { announceOnly } from '#/announce'
import { createFileRoute, notFound } from '@tanstack/react-router'
import { ArrowLeft, Chevron, button } from '#/components/Button'
import { Link } from '#/components/Link'
import { NewsList } from '#/components/NewsList'
import { Body } from '#/components/Prose'
import { focus, frame, link, text } from '#/components/styles'
import { NAME, SITE_URL } from '#/data/site'
import { formatDate, posts } from '#/news'
import { colors, sizes, space, type } from '../../tokens.stylex'

export const Route = createFileRoute('/news/$slug')({
  beforeLoad: announceOnly,
  loader: ({ params }) => {
    const post = posts.find((p) => p.slug === params.slug)
    if (!post) throw notFound()
    // Only what the head needs crosses to the browser; the page finds the post again by slug.
    return { slug: post.slug, title: post.title, summary: post.summary, date: post.date }
  },
  head: ({ loaderData }) =>
    loaderData
      ? {
          meta: [
            { title: `${loaderData.title} · ${NAME}` },
            { name: 'description', content: loaderData.summary },
            { property: 'og:title', content: loaderData.title },
            { property: 'og:description', content: loaderData.summary },
            { property: 'og:type', content: 'article' },
            { property: 'og:url', content: `${SITE_URL}/news/${loaderData.slug}` },
            { property: 'article:published_time', content: loaderData.date },
          ],
        }
      : {},
  component: PostPage,
})

const styles = stylex.create({
  top: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: space.x4, paddingTop: space.x12 },
  back: { display: 'inline-flex', alignItems: 'center', gap: space.x2, fontSize: type.small, lineHeight: '20px' },
  meta: { display: 'flex', gap: space.x3, fontSize: type.small, lineHeight: '20px' },
  category: { color: colors.fg, fontWeight: 500 },
  head: { display: 'grid', justifyItems: 'center', gap: space.x6, textAlign: 'center', paddingTop: space.x12, paddingBottom: space.x12 },
  title: { maxWidth: '16em' },
  lede: { maxWidth: '36em' },
  end: { display: 'grid', gap: space.x6, marginTop: space.x16, paddingTop: space.x10, borderTopWidth: sizes.hair, borderTopStyle: 'solid', borderTopColor: colors.line },
  endRow: { display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', justifyContent: 'space-between', gap: space.x4 },
  actions: { display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: space.x3 },
})

/** A post: a row with the way back and the date, the title and a paragraph centred, then one
 * reading column. Figures marked `wide` break out of the column to the width of the title row.
 * Every later announcement uses this page: it is a file in src/news, nothing here changes. */
function PostPage() {
  const { slug } = Route.useParams()
  const post = posts.find((p) => p.slug === slug)
  if (!post) return null
  const others = posts.filter((p) => p.slug !== post.slug).slice(0, 3)
  return (
    <article>
      <div {...stylex.props(frame.wide, styles.top)}>
        <Link to="/news" {...stylex.props(link.quiet, styles.back, focus.ring)}>
          <ArrowLeft />
          Back to news
        </Link>
        <p {...stylex.props(text.muted, styles.meta)}>
          <span {...stylex.props(styles.category)}>{post.category}</span>
          <time dateTime={post.date}>{formatDate(post.date)}</time>
        </p>
      </div>
      <header {...stylex.props(frame.wide, styles.head)}>
        <h1 {...stylex.props(text.h1, styles.title)}>{post.title}</h1>
        <p {...stylex.props(text.lede, styles.lede)}>{post.lede}</p>
        <div {...stylex.props(styles.actions)}>
          <Link to="/docs" {...button('primary')}>
            Read the docs
            <Chevron />
          </Link>
          <Link to="/benchmarks" {...button('secondary')}>
            See the benchmarks
          </Link>
        </div>
      </header>
      <div {...stylex.props(frame.prose)}>
        <Body>
          <post.Body />
        </Body>
        {others.length > 0 && (
          <aside aria-labelledby="more-h" {...stylex.props(styles.end)}>
            <div {...stylex.props(styles.endRow)}>
              <h2 id="more-h" {...stylex.props(text.h3)}>
                More news
              </h2>
              <Link to="/news" {...stylex.props(link.more, focus.ring)}>
                All posts
                <Chevron />
              </Link>
            </div>
            <NewsList posts={others} />
          </aside>
        )}
      </div>
    </article>
  )
}

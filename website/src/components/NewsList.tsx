import * as stylex from '@stylexjs/stylex'
import { formatDate, type Post } from '#/news'
import { colors, media, sizes, space } from '../tokens.stylex'
import { Link } from './Link'
import { focus, text } from './styles'

/* Posts as rows under hairlines: title and one line of summary at the left, category and date
   at the right. The home page shows the latest few; the news page all of them. */

const styles = stylex.create({
  list: { display: 'grid', borderTopWidth: sizes.hair, borderTopStyle: 'solid', borderTopColor: colors.line },
  row: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 1fr) auto', [media.phone]: 'minmax(0, 1fr)' },
    alignItems: 'center',
    columnGap: space.x8,
    rowGap: space.x2,
    paddingBlock: space.x5,
    borderBottomWidth: sizes.hair,
    borderBottomStyle: 'solid',
    borderBottomColor: colors.line,
    textDecorationLine: 'none',
    transitionProperty: 'opacity',
    transitionDuration: '150ms',
    opacity: { default: 1, ':hover': 0.72 },
  },
  words: { display: 'grid', gap: space.x1, minWidth: 0 },
  summary: { overflow: 'hidden', display: '-webkit-box', WebkitBoxOrient: 'vertical', WebkitLineClamp: 2 },
  meta: { display: 'flex', gap: space.x3, whiteSpace: 'nowrap' },
  category: { color: colors.fg, fontWeight: 500 },
})

export function NewsList({ posts }: { posts: Post[] }) {
  return (
    <ol {...stylex.props(styles.list)}>
      {posts.map((p) => (
        <li key={p.slug}>
          <Link to="/news/$slug" params={{ slug: p.slug }} {...stylex.props(styles.row, focus.ring)}>
            <span {...stylex.props(styles.words)}>
              <span {...stylex.props(text.title)}>{p.title}</span>
              <span {...stylex.props(text.small, styles.summary)}>{p.summary}</span>
            </span>
            <span {...stylex.props(text.micro, styles.meta)}>
              <span {...stylex.props(styles.category)}>{p.category}</span>
              <time dateTime={p.date}>{formatDate(p.date)}</time>
            </span>
          </Link>
        </li>
      ))}
    </ol>
  )
}

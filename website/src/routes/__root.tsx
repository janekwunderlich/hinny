import * as stylex from '@stylexjs/stylex'
import { HeadContent, Scripts, createRootRoute } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import { button } from '#/components/Button'
import { DevStyles } from '#/components/DevStyles'
import { Footer } from '#/components/Footer'
import { Link } from '#/components/Link'
import { Hinny } from '#/components/Mascot'
import { TopBar } from '#/components/TopBar'
import { frame, text } from '#/components/styles'
import { ANNOUNCE, NAME, SITE_URL } from '#/data/site'
import { colors, fonts, radii, space, type } from '../tokens.stylex'

import appCss from '../styles.css?url'

const title = `${NAME}: embedded hybrid search`
const description = `${ANNOUNCE ? 'Alpha. ' : ''}BM25 full text, dense vectors, learned sparse terms, and attribute filters in one embedded Rust database. Local memory-mapped files, instant writes, Rust/Python/Node.js/C bindings.`

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1, viewport-fit=cover' },
      { title },
      { name: 'description', content: description },
      { name: 'theme-color', content: '#ffffff', media: '(prefers-color-scheme: light)' },
      { name: 'theme-color', content: '#0a0a0a', media: '(prefers-color-scheme: dark)' },
      // The card a shared link shows. A page sets its own title, description and address
      // (seo.ts); the image is one for the whole site, and the announcement has its own.
      { property: 'og:site_name', content: NAME },
      { property: 'og:title', content: title },
      { property: 'og:description', content: description },
      { property: 'og:type', content: 'website' },
      { property: 'og:image', content: `${SITE_URL}/${ANNOUNCE ? 'og-soon.png' : 'og.png'}` },
      { property: 'og:image:width', content: '1200' },
      { property: 'og:image:height', content: '630' },
      { property: 'og:image:alt', content: `The ${NAME} mark, a hinny's head, over the name and the line “An embedded hybrid search database${ANNOUNCE ? '. Coming soon.' : ''}”` },
      { name: 'twitter:card', content: 'summary_large_image' },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
      { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' },
      { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32.png' },
      { rel: 'apple-touch-icon', href: '/apple-touch-icon.png' },
      // Geist (OFL 1.1, public/fonts), served from here: no third-party font requests.
      { rel: 'preload', href: '/fonts/Geist-Variable.woff2', as: 'font', type: 'font/woff2', crossOrigin: 'anonymous' },
    ],
  }),
  shellComponent: RootDocument,
  notFoundComponent: NotFound,
})

const styles = stylex.create({
  html: { backgroundColor: colors.bg },
  selection: { '::selection': { backgroundColor: colors.fg, color: colors.bg } },
  body: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: colors.bg,
    color: colors.text,
    fontFamily: fonts.sans,
    fontSize: type.body,
    lineHeight: 1.5,
    fontSynthesis: 'none',
    WebkitFontSmoothing: 'antialiased',
    MozOsxFontSmoothing: 'grayscale',
    textRendering: 'optimizeLegibility',
    overflowX: 'clip',
  },
  skip: {
    position: 'absolute',
    left: { default: '-9999px', ':focus': space.gutter },
    top: space.x2,
    zIndex: 200,
    backgroundColor: colors.fg,
    color: colors.bg,
    paddingBlock: space.x2,
    paddingInline: space.x3,
    borderRadius: radii.small,
    fontSize: type.small,
  },
  main: { flexGrow: 1, minWidth: 0 },
  missing: { display: 'grid', gap: space.x6, justifyItems: 'start', paddingBlock: space.section },
})

function RootDocument({ children }: { children: ReactNode }) {
  return (
    <html lang="en" {...stylex.props(styles.html)}>
      <head>
        <HeadContent />
        <DevStyles />
      </head>
      <body {...stylex.props(styles.body, styles.selection)}>
        <a href="#content" {...stylex.props(styles.skip)}>
          Skip to the content
        </a>
        <TopBar />
        <main id="content" {...stylex.props(styles.main)}>
          {children}
        </main>
        <Footer />
        <Scripts />
      </body>
    </html>
  )
}

function NotFound() {
  return (
    <section {...stylex.props(frame.page, styles.missing)}>
      <Hinny size={96} coat="rose" />
      <h1 {...stylex.props(text.h1)}>Nothing here</h1>
      <p {...stylex.props(text.lede)}>That page doesn't exist.</p>
      <Link to="/" {...button('primary')}>
        Go to the start
      </Link>
    </section>
  )
}

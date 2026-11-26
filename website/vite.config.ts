import { defineConfig } from 'vite'
import { tanstackStart } from '@tanstack/react-start/plugin/vite'
import { nitro } from 'nitro/vite'
import viteReact from '@vitejs/plugin-react'
import stylex from '@stylexjs/unplugin'
import { Features } from 'lightningcss'

// Pages are prerendered to HTML at build time (the site has no server work to do); Nitro builds
// the output for wherever it runs, Vercel's format on Vercel.
export default defineConfig({
  resolve: { tsconfigPaths: true },
  plugins: [
    // StyleX compiles the styles in the components to atomic CSS and appends it to the one CSS
    // file the app imports (src/styles.css). It goes before the framework plugins, as its
    // documentation asks, so Fast Refresh keeps working.
    stylex.vite({
      // StyleX's rules go in cascade layers after `reset`, the layer of src/styles.css.
      useCSSLayers: { before: ['reset'] },
      dev: process.env.NODE_ENV === 'development',
      runtimeInjection: false,
      // The client build and the server build each collect the rules of what they compile; kept on
      // disk in development, so both see all of them.
      devPersistToDisk: true,
      // The colour tokens are light-dark() values, which follow the page's colour scheme (and so
      // the switch in the footer). Left as written: lowered for old browsers, they would follow
      // only the system's setting. Chrome 123, Safari 17.5 and Firefox 120 understand them.
      lightningcssOptions: { exclude: Features.LightDark },
    }),
    tanstackStart({
      // With VITE_ANNOUNCE=1 the site is one page (src/data/site.ts, ANNOUNCE): only it is built.
      prerender: { enabled: true, crawlLinks: true, failOnError: true, filter: ({ path }) => !path.includes('#') && (process.env.VITE_ANNOUNCE !== '1' || path === '/') },
    }),
    nitro(),
    viteReact(),
  ],
})

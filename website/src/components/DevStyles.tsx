import { useEffect } from 'react'

/** In development, StyleX's Vite plugin serves the CSS it has compiled so far from a virtual
 * stylesheet and reloads it after an edit; this links it and starts that reloading. In a build
 * there is nothing to do: the plugin has appended the CSS to the app's own stylesheet. This is
 * the arrangement StyleX's Vite guide gives for apps whose entry is a component, not an HTML file. */
export function DevStyles() {
  useEffect(() => {
    if (import.meta.env.DEV) void import('virtual:stylex:runtime')
  }, [])
  return import.meta.env.DEV ? <link rel="stylesheet" href="/virtual:stylex.css" /> : null
}

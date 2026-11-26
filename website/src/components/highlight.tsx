import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'
import { colors, syntax } from '../tokens.stylex'

/* Syntax colours for the code on the site. Code is written as plain text and coloured here by a
   small tokenizer: comments, strings, numbers, keywords and calls. It knows just enough of
   Python, Rust, C and a shell prompt for the site's own samples; it is not a parser. */

export type Lang = 'python' | 'rust' | 'js' | 'c' | 'shell'

const KEYWORDS: Record<Lang, Set<string>> = {
  python: new Set(['import', 'from', 'for', 'in', 'if', 'else', 'return', 'def', 'as', 'with', 'not', 'and', 'or', 'None', 'True', 'False', 'lambda']),
  rust: new Set(['let', 'use', 'fn', 'pub', 'mut', 'for', 'in', 'if', 'else', 'match', 'return', 'struct', 'impl', 'as']),
  js: new Set(['const', 'let', 'var', 'new', 'import', 'from', 'for', 'of', 'in', 'if', 'else', 'return', 'async', 'await', 'true', 'false', 'null']),
  c: new Set(['char', 'int', 'float', 'const', 'size_t', 'uint64_t', 'void', 'return', 'if', 'else', 'struct', 'null', 'NULL']),
  shell: new Set(),
}

const COMMENT: Record<Lang, string> = {
  python: '#[^\\n]*',
  rust: '\\/\\/[^\\n]*',
  js: '\\/\\*[\\s\\S]*?\\*\\/|\\/\\/[^\\n]*',
  c: '\\/\\*[\\s\\S]*?\\*\\/|\\/\\/[^\\n]*',
  shell: '(?:^|(?<=\\n))\\$ ',
}

const styles = stylex.create({
  comment: { color: colors.muted, fontStyle: 'italic' },
  prompt: { color: colors.muted, userSelect: 'none' },
  string: { color: syntax.string },
  number: { color: syntax.number },
  keyword: { color: syntax.keyword },
  call: { color: syntax.call },
})

/** `code` as spans coloured for `lang`. */
export function highlight(code: string, lang: Lang): ReactNode[] {
  // In order: a comment (or the shell's prompt), a string, a number, a name before "(", any other name.
  const rule = new RegExp(`(${COMMENT[lang]})|(f?"(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*')|(\\b\\d+(?:\\.\\d+)?\\b)|([A-Za-z_][A-Za-z0-9_]*)(?=\\()|([A-Za-z_][A-Za-z0-9_]*)`, 'g')
  const out: ReactNode[] = []
  let last = 0
  let key = 0
  for (const m of code.matchAll(rule)) {
    const [text, comment, string, number, call, name] = m
    const style = comment ? (lang === 'shell' ? styles.prompt : styles.comment) : string ? styles.string : number ? styles.number : call ? (KEYWORDS[lang].has(call) ? styles.keyword : styles.call) : name && KEYWORDS[lang].has(name) ? styles.keyword : null
    if (!style) continue
    if (m.index > last) out.push(code.slice(last, m.index))
    out.push(
      <span key={key++} {...stylex.props(style)}>
        {text}
      </span>,
    )
    last = m.index + text.length
  }
  if (last < code.length) out.push(code.slice(last))
  return out
}

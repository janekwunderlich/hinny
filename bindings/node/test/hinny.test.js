'use strict'

// hinny from Node: schemas, documents, text, vector and hybrid queries, filters, snapshots,
// batches, reopening, errors, 64-bit keys. Run with `npm test` (node --test) after `npm run build`.

const { test, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawnSync } = require('node:child_process')

const hinny = require('..')
const { Database, attr, key } = hinny

const SCHEMA = { text: { title: 2, body: 1 }, ints: ['year'], strings: ['kind'], vector: 3 }

let dir, db, ns

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hinny-node-'))
  db = new Database(dir)
  ns = db.namespace('docs', SCHEMA)
  ns.upsertMany([
    { key: 1, text: { title: 'apartment lease', body: 'the lease for the apartment' }, attrs: { year: 2025, kind: 'pdf' }, vector: [0, 0, 1] },
    { key: 2, text: { title: 'holiday photos', body: 'beach' }, attrs: { year: 2024, kind: 'jpg' }, vector: new Float32Array([1, 0, 0]) },
    { key: 3, text: { title: 'lease renewal', body: 'renewing it' }, attrs: { year: 2026, kind: 'pdf' }, vector: new Float64Array([0.9, 0.1, 0]) },
    { key: 4, text: { title: 'groceries', body: 'milk' }, attrs: { year: 2023, kind: 'txt' }, vector: [0, 1, 0] },
  ])
})

afterEach(() => {
  db.close()
  fs.rmSync(dir, { recursive: true, force: true })
})

const keys = (hits) => hits.map((h) => Number(h.key))

/** Asserts that `f` throws an Error with this code (and a message matching `message`). */
function throws(f, code, message) {
  assert.throws(f, (e) => {
    assert.ok(e instanceof Error, `not an Error: ${e}`)
    assert.equal(e.code, code, `${e.code}: ${e.message}`)
    if (message) assert.match(e.message, message)
    return true
  })
}

test('text, vector and hybrid queries', () => {
  const text = ns.query({ text: 'apartment lease' })
  assert.deepEqual(keys(text).slice(0, 2), [1, 3])
  assert.equal(typeof text[0].key, 'bigint')
  assert.equal(typeof text[0].textScore, 'number')
  assert.equal(text[0].vectorScore, null)

  for (const v of [new Float32Array([1, 0, 0]), new Float64Array([1, 0, 0]), [1, 0, 0]]) {
    const near = ns.query({ vector: v, limit: 2 })
    assert.deepEqual(keys(near), [2, 3])
    assert.ok(Math.abs(near[0].score - 1) < 1e-5)
    assert.equal(near[0].textScore, null)
  }

  // The best of each retriever, and 3, good at both.
  const both = ns.query({ text: 'lease', vector: [1, 0, 0], limit: 3 })
  assert.deepEqual(new Set(keys(both)), new Set([1, 2, 3]))
  assert.ok(keys(ns.query({ text: 'lease', vector: [1, 0, 0], limit: 3, fusion: 'rrf' })).slice(0, 2).includes(3))
  const three = both.find((h) => h.key === 3n)
  assert.equal(typeof three.textScore, 'number')
  assert.equal(typeof three.vectorScore, 'number')
  assert.deepEqual(three.attrs, { year: 2026, kind: 'pdf' })

  // All words, and listing without a query.
  assert.deepEqual(keys(ns.query({ text: 'apartment renewal', allWords: true })), [])
  assert.deepEqual(keys(ns.query({ text: 'lease renewal', allWords: true })), [3])
  assert.equal(ns.query().length, 4)
  assert.equal(ns.query({ limit: 2 }).length, 2)
  assert.equal(ns.query({ vector: [1, 0, 0], nprobe: 4 })[0].key, 2n)
})

test('filters on integers', () => {
  for (const where of ['memory', 'segment']) {
    const q = (filter, orderBy = 'year') => keys(ns.query({ filter, orderBy }))
    assert.deepEqual(q(attr('year').eq(2025)), [1], where)
    assert.deepEqual(q(attr('year').ne(2025)), [4, 2, 3], where)
    assert.deepEqual(q(attr('year').gte(2025)), [1, 3], where)
    assert.deepEqual(q(attr('year').gt(2025)), [3], where)
    assert.deepEqual(q(attr('year').lt(2025)), [4, 2], where)
    assert.deepEqual(q(attr('year').lte(2025)), [4, 2, 1], where)
    assert.deepEqual(q(attr('year').between(2024, 2025)), [2, 1], where)
    assert.deepEqual(q(attr('year').oneOf([2023, 2026, 1999])), [4, 3], where)
    assert.deepEqual(q(attr('year').exists()), [4, 2, 1, 3], where)
    // A fraction asked of integers: the integers in the range. Bigints are integers too.
    assert.deepEqual(q(attr('year').between(2023.5, 2025.5)), [2, 1], where)
    assert.deepEqual(q(attr('year').gt(2024.5)), [1, 3], where)
    assert.deepEqual(q(attr('year').gte(2025n)), [1, 3], where)
    // Combined.
    assert.deepEqual(q(attr('kind').eq('pdf').and(attr('year').lt(2026).not())), [3], where)
    assert.deepEqual(q(attr('year').lt(2024).or(attr('year').gt(2025))), [4, 3], where)
    assert.deepEqual(q(attr('year').gte(2024).and(attr('year').lte(2025)).not()), [4, 3], where)
    ns.flush()
  }
})

test('tags', () => {
  const tg = db.namespace('tagged', { text: 'body', tags: ['labels'], strings: ['owner'] })
  assert.deepEqual(tg.schema.tags, ['labels'])
  tg.upsertMany([
    { key: 1, text: 'tax return', attrs: { labels: ['tax', '2025', 'tax'], owner: 'mia' } },
    { key: 2, text: 'tax letter', attrs: { labels: ['tax', 'Inbox'] } },
    { key: 3, text: 'holiday', attrs: { labels: ['family'] } },
    { key: 4, text: 'nothing', attrs: { labels: [] } },
  ])
  for (const where of ['memory', 'segment']) {
    const q = (filter) => keys(tg.query({ filter })).sort()
    // A set: each once, in order.
    assert.deepEqual(tg.get(1), { labels: ['2025', 'tax'], owner: 'mia' }, where)
    assert.deepEqual(tg.get(4), {}, where)
    assert.deepEqual(q(attr('labels').has('tax')), [1, 2], where)
    assert.deepEqual(q(attr('labels').eq('tax')), [1, 2], where)
    assert.deepEqual(q(attr('labels').ne('tax')), [3, 4], where)
    assert.deepEqual(q(attr('labels').hasAny(['family', '2025'])), [1, 3], where)
    assert.deepEqual(q(attr('labels').hasAll(['tax', 'Inbox'])), [2], where)
    assert.deepEqual(q(attr('labels').prefix('in')), [2], where)
    assert.deepEqual(q(attr('labels').exists().not()), [4], where)
    assert.deepEqual(keys(tg.query({ text: 'tax', filter: attr('labels').has('Inbox') })), [2], where)
    assert.deepEqual(tg.query({ text: 'letter' })[0].attrs.labels, ['Inbox', 'tax'], where)
    assert.equal(tg.count(attr('labels').has('tax')), 2, where)
    assert.deepEqual(tg.countBy('labels', null, 1), [['tax', 2]], where)
    tg.flush()
  }
  assert.throws(() => tg.upsert({ key: 5, attrs: { labels: [1, 2] } }), { code: 'ERR_HINNY_INVALID' })
  assert.throws(() => tg.upsert({ key: 5, attrs: { owner: ['a', 'b'] } }), { code: 'ERR_HINNY_INVALID' })
  tg.close()
})

test('filters on floats', () => {
  const sc = db.namespace('scored', { text: ['body'], floats: ['score'], ints: ['n'] })
  assert.deepEqual(sc.schema.floats, ['score'])
  sc.upsertMany([
    { key: 1, text: 'one', attrs: { score: 0.25, n: 1 } },
    { key: 2, text: 'two', attrs: { score: 2, n: 2 } }, // an integer for a float attribute is that number
    { key: 3, text: 'three', attrs: { score: -1.5, n: 3 } },
    { key: 4, text: 'four', attrs: { n: 4 } },
  ])
  for (const where of ['memory', 'segment']) {
    const q = (filter, orderBy) => keys(sc.query({ filter, orderBy }))
    assert.deepEqual(sc.get(1), { score: 0.25, n: 1 }, where)
    assert.equal(sc.get(2).score, 2, where)
    assert.deepEqual(q(attr('score').gte(0.25), '-score'), [2, 1], where)
    assert.deepEqual(q(attr('score').gt(0.25)), [2], where)
    assert.deepEqual(q(attr('score').lt(0), 'score'), [3], where)
    assert.deepEqual(q(attr('score').lte(0.25), 'score'), [3, 1], where)
    assert.deepEqual(q(attr('score').between(-2, 0.5), 'score'), [3, 1], where)
    assert.deepEqual(q(attr('score').eq(2)), [2], where)
    assert.deepEqual(q(attr('score').ne(2), 'n'), [1, 3, 4], where)
    assert.deepEqual(q(attr('score').oneOf([0.25, -1.5, 7]), 'score'), [3, 1], where)
    assert.deepEqual(q(attr('score').exists().not()), [4], where)
    assert.deepEqual(keys(sc.query({ text: 'one two three four', filter: attr('score').lte(0.25), orderBy: 'score' })), [3, 1], where)
    assert.deepEqual(sc.countBy('score', attr('n').lte(2)), [[0.25, 1], [2, 1]], where)
    sc.flush()
  }
})

test('filters on strings', () => {
  const s = db.namespace('files', { text: 'name', strings: ['path', 'kind'] })
  s.upsertMany([
    { key: 1, text: 'lease', attrs: { path: 'Documents/Lease 2025.pdf', kind: 'pdf' } },
    { key: 2, text: 'photo', attrs: { path: 'Pictures/beach.jpg', kind: 'jpg' } },
    { key: 3, text: 'notes', attrs: { path: 'Documents/notes.txt' } },
  ])
  for (const where of ['memory', 'segment']) {
    const q = (filter) => keys(s.query({ filter })).sort()
    assert.deepEqual(q(attr('kind').eq('pdf')), [1], where)
    assert.deepEqual(q(attr('kind').ne('pdf')), [2, 3], where)
    assert.deepEqual(q(attr('kind').oneOf(['jpg', 'txt'])), [2], where)
    assert.deepEqual(q(attr('path').prefix('documents/')), [1, 3], where) // ignoring case
    assert.deepEqual(q(attr('path').contains('LEASE')), [1], where)
    assert.deepEqual(q(attr('kind').exists()), [1, 2], where)
    assert.deepEqual(q(attr('path').prefix('Documents/').and(attr('kind').exists().not())), [3], where)
    assert.deepEqual(q(attr('kind').eq('jpg').or(attr('path').contains('notes'))), [2, 3], where)
    s.flush()
  }
})

test('patterns', () => {
  const f = db.namespace('patterns', { text: 'name', strings: ['path'], tags: ['labels'] })
  f.upsertMany([
    { key: 1, text: 'taxes', attrs: { path: '/home/mia/taxes/2025.pdf', labels: ['Tax', 'inbox'] } },
    { key: 2, text: 'taxes', attrs: { path: '/home/leo/taxes/2024.PDF', labels: ['family'] } },
    { key: 3, text: 'photo', attrs: { path: '/home/mia/photos/2025.pdf.jpg' } },
  ])
  for (const where of ['memory', 'segment']) {
    const q = (filter) => keys(f.query({ filter })).sort()
    assert.deepEqual(q(attr('path').glob('/home/*/taxes/*.pdf')), [1], where)
    assert.deepEqual(q(attr('path').iglob('/home/*/taxes/*.pdf')), [1, 2], where)
    assert.deepEqual(q(attr('path').glob('/home/???/taxes/20??.pdf')), [1], where)
    assert.deepEqual(q(attr('path').glob('*.pdf')), [1], where) // the whole string
    assert.deepEqual(q(attr('path').glob('*.pdf*').and(attr('labels').exists().not())), [3], where)
    assert.deepEqual(q(attr('labels').glob('T?x')), [1], where)
    assert.deepEqual(q(attr('labels').glob('t?x')), [], where)
    assert.deepEqual(q(attr('labels').iglob('t?x')), [1], where)
    f.flush()
  }
})

test('ordering', () => {
  assert.deepEqual(keys(ns.query({ filter: attr('year').gte(2024), orderBy: '-year' })), [3, 1, 2])
  assert.deepEqual(keys(ns.query({ orderBy: 'year' })), [4, 2, 1, 3])
  assert.deepEqual(keys(ns.query({ text: 'lease', orderBy: '-year' })), [3, 1])
  assert.deepEqual(keys(ns.query({ orderBy: '-year', limit: 2 })), [3, 1])
})

test('counts', () => {
  assert.equal(ns.count(), 4)
  assert.equal(ns.size, 4)
  assert.equal(ns.count(attr('kind').eq('pdf')), 2)
  assert.equal(ns.count(attr('year').gt(2030)), 0)
  assert.deepEqual(ns.countBy('kind'), [['pdf', 2], ['jpg', 1], ['txt', 1]])
  assert.deepEqual(ns.countBy('kind', null, 1), [['pdf', 2]])
  assert.deepEqual(ns.countBy('year', attr('kind').eq('pdf')), [[2025, 1], [2026, 1]])
  assert.deepEqual(ns.countBy('nothing'), [])
  // The same from segments, and a snapshot keeps its own.
  const before = ns.snapshot()
  ns.flush()
  ns.delete(1)
  assert.equal(ns.count(), 3)
  assert.deepEqual(ns.countBy('kind'), [['jpg', 1], ['pdf', 1], ['txt', 1]])
  assert.equal(before.count(), 4)
  assert.deepEqual(before.countBy('kind', undefined, 1), [['pdf', 2]])
})

test('documents: get, text, delete, versions, schema', () => {
  assert.deepEqual(ns.get(2), { year: 2024, kind: 'jpg' })
  assert.deepEqual(ns.text(2), { title: 'holiday photos', body: 'beach' })
  assert.equal(ns.get(99), null)
  assert.equal(ns.text(99), null)
  ns.delete(2)
  assert.equal(ns.get(2), null)
  assert.ok(!keys(ns.query({ vector: [1, 0, 0] })).includes(2))
  ns.upsert({ key: key('doc-17'), text: { title: 'string ids' }, version: 7 })
  const versions = ns.versions()
  assert.ok(versions instanceof Map)
  assert.equal(versions.get(key('doc-17')), 7n)
  assert.equal(versions.get(1n), 0n)
  assert.equal(ns.name, 'docs')
  assert.deepEqual(ns.schema, { text: { title: 2, body: 1 }, sparse: [], ints: ['year'], floats: [], strings: ['kind'], tags: [], vector: 3, metric: 'cosine', storeText: true })
  assert.deepEqual(db.namespaces(), ['docs'])
  assert.equal(db.path, dir)
  // Undefined and null attributes are left out; replacing a document replaces all of it.
  ns.upsert({ key: 1, text: { title: 'replaced' }, attrs: { year: 2000, kind: undefined } })
  assert.deepEqual(ns.get(1), { year: 2000 })
  // Iterables, not only arrays.
  function* more() {
    for (let i = 10; i < 13; i++) yield { key: i, text: { title: `more ${i}` } }
  }
  assert.equal(ns.upsertMany(more()), 3)
  assert.equal(ns.deleteMany(new Set([10, 11])), 2)
  assert.deepEqual([...ns.versions().keys()].filter((k) => k >= 10n && k < 13n), [12n])
})

test('keys above 2^53, as bigints', () => {
  const big = [2n ** 53n + 1n, 2n ** 63n + 7n, 2n ** 64n - 1n]
  ns.upsertMany(big.map((k, i) => ({ key: k, text: { title: `huge lease ${i}` }, attrs: { year: 3000 + i }, version: k })))
  for (const k of big) assert.notEqual(ns.get(k), null)
  // Neighbours that would round to the same double are different documents.
  assert.equal(ns.get(2n ** 53n), null)
  assert.equal(ns.get(2n ** 64n - 2n), null)
  const hits = ns.query({ filter: attr('year').gte(3000), orderBy: 'year' })
  assert.deepEqual(hits.map((h) => h.key), big)
  const versions = ns.versions()
  for (const k of big) assert.equal(versions.get(k), k)
  assert.deepEqual(ns.snapshot().query({ text: 'huge', limit: 10 }).map((h) => h.key).sort(), [...big].sort())
  ns.flush()
  assert.deepEqual(ns.query({ filter: attr('year').gte(3000), orderBy: 'year' }).map((h) => h.key), big)
  assert.equal(ns.apply({ delete: [big[2]] }), 1)
  ns.delete(big[1])
  assert.equal(ns.deleteMany([big[0]]), 1)
  assert.equal(ns.count(attr('year').gte(3000)), 0)
  // A number above 2^53 - 1 may already be rounded: it's refused, with the remedy.
  throws(() => ns.get(2 ** 60), 'ERR_HINNY_INVALID', /bigint/)
  throws(() => ns.upsert({ key: 2 ** 53 }), 'ERR_HINNY_INVALID', /bigint/)
  throws(() => ns.get(-1), 'ERR_HINNY_INVALID')
  throws(() => ns.get(1.5), 'ERR_HINNY_INVALID')
  throws(() => ns.get(-1n), 'ERR_HINNY_INVALID')
  throws(() => ns.get(2n ** 64n), 'ERR_HINNY_INVALID')
  throws(() => ns.get('1'), 'ERR_HINNY_INVALID')
  // Integer attributes beyond 2^53 come back as bigints, exactly; safe ones as numbers.
  const at = db.namespace('at', { text: 'body', ints: ['t'] })
  at.upsertMany([
    { key: 1, text: 'a', attrs: { t: 2n ** 62n + 3n } },
    { key: 2, text: 'b', attrs: { t: -(2n ** 63n) + 1n } },
    { key: 3, text: 'c', attrs: { t: Number.MAX_SAFE_INTEGER } },
  ])
  assert.equal(at.get(1).t, 2n ** 62n + 3n)
  assert.equal(at.get(2).t, -(2n ** 63n) + 1n)
  assert.equal(at.get(3).t, Number.MAX_SAFE_INTEGER)
  assert.deepEqual(keys(at.query({ filter: attr('t').gt(2n ** 62n) })), [1])
  throws(() => at.upsert({ key: 4, attrs: { t: 2n ** 63n } }), 'ERR_HINNY_INVALID')
  // -2^63 marks a missing integer in the engine, so it's refused rather than dropped.
  throws(() => at.upsert({ key: 4, attrs: { t: -(2n ** 63n) } }), 'ERR_HINNY_INVALID', /reserved/)
})

test('key() matches FNV-1a over UTF-8, in 63 bits', () => {
  const fnv = (s) => {
    let h = 0xcbf29ce484222325n
    for (const b of Buffer.from(s, 'utf8')) h = ((h ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn
    return h & (2n ** 63n - 1n)
  }
  for (const id of ['', 'doc-17', 'handbook/intro.md#3', 'Grüße 😀']) {
    assert.equal(key(id), fnv(id))
    assert.ok(key(id) < 2n ** 63n)
  }
  assert.equal(key('doc-17'), key('doc-17'))
  assert.notEqual(key('doc-17'), key('doc-18'))
  throws(() => key(17), 'ERR_HINNY_INVALID')
})

test('snapshots keep their view through later writes', () => {
  const snap = ns.snapshot()
  ns.upsert({ key: 99, text: { title: 'Later lease' }, attrs: { year: 2030 } })
  ns.delete(1)
  ns.flush()
  ns.upsert({ key: 98, text: { title: 'Even later lease' } })
  ns.optimize()
  assert.deepEqual([...snap.versions().keys()].sort(), [1n, 2n, 3n, 4n])
  assert.deepEqual(snap.get(1), { year: 2025, kind: 'pdf' })
  assert.equal(snap.text(1).title, 'apartment lease')
  assert.equal(snap.get(99), null)
  assert.deepEqual(keys(snap.query({ text: 'lease', limit: 10 })).sort(), [1, 3])
  assert.equal(snap.explain({ text: 'lease' }).hits.length, 2)
  assert.equal(snap.count(), 4)
  assert.ok(ns.snapshot().seq > snap.seq)
  assert.equal(typeof snap.seq, 'bigint')
  // Now.
  assert.equal(ns.get(1), null)
  assert.deepEqual(keys(ns.query({ text: 'lease', limit: 10 })).sort(), [3, 98, 99])
  snap.close()
  throws(() => snap.count(), 'ERR_HINNY_CLOSED')
})

test('apply is one batch: all or nothing', () => {
  const before = ns.snapshot()
  assert.equal(ns.apply({ upsert: [{ key: 300, text: { title: 'moved lease' } }], delete: [3, 99] }), 3)
  assert.equal(ns.get(3), null)
  assert.notEqual(ns.get(300), null)
  assert.notEqual(before.get(3), null)
  assert.equal(before.get(300), null)
  assert.equal(ns.deleteMany([2, 4]), 2)
  assert.deepEqual([...ns.versions().keys()].sort(), [1n, 300n])
  // A batch with a bad document changes nothing: neither its good documents nor its deletes.
  throws(() => ns.apply({ upsert: [{ key: 200, text: { title: 'ok' } }, { key: 201, text: { nofield: 'x' } }], delete: [1] }), 'ERR_HINNY_INVALID', /nofield/)
  throws(() => ns.upsertMany([{ key: 200, text: { title: 'ok' } }, { key: 201, vector: [1, 2] }]), 'ERR_HINNY_INVALID')
  throws(() => ns.apply({ upsert: [{ key: 200, text: { title: 'ok' } }], delete: [1, -5] }), 'ERR_HINNY_INVALID')
  assert.equal(ns.get(200), null)
  assert.notEqual(ns.get(1), null)
  assert.deepEqual([...ns.versions().keys()].sort(), [1n, 300n])
  throws(() => ns.apply({ upsert: [], remove: [1] }), 'ERR_HINNY_INVALID', /remove/)
})

test('reopening a database', () => {
  const q = { text: 'lease', vector: [1, 0, 0], limit: 3 }
  const before = keys(ns.query(q))
  // One handle at a time: the directory is locked while this one is open.
  throws(() => new Database(dir).namespace('docs', SCHEMA), 'ERR_HINNY_LOCKED')
  ns.close()
  assert.equal(ns.closed, true)
  throws(() => ns.query(q), 'ERR_HINNY_CLOSED')
  const db2 = new Database(dir)
  const again = db2.namespace('docs', SCHEMA)
  assert.deepEqual(keys(again.query(q)), before)
  assert.deepEqual(again.get(3), { year: 2026, kind: 'pdf' })
  again.flush()
  // Closing the database closes its namespaces and their snapshots.
  const snap = again.snapshot()
  db2.close()
  throws(() => again.size, 'ERR_HINNY_CLOSED')
  throws(() => snap.count(), 'ERR_HINNY_CLOSED')
  throws(() => db2.namespace('docs', SCHEMA), 'ERR_HINNY_CLOSED')
  {
    using third = new Database(dir).namespace('docs', SCHEMA)
    assert.deepEqual(keys(third.query(q)), before)
  }
  // `using` closed it.
  const fourth = db.namespace('docs', SCHEMA)
  assert.equal(fourth.size, 4)
  // A schema can grow, but not change.
  fourth.close()
  const grown = db.namespace('docs', { ...SCHEMA, floats: ['rating'] })
  assert.deepEqual(grown.schema.floats, ['rating'])
  grown.close()
  throws(() => db.namespace('docs', { text: { body: 1, title: 2 }, vector: 3 }), 'ERR_HINNY_INVALID', /another schema/)
  // As it is, without restating the schema; and not what doesn't exist.
  const asItIs = db.open('docs')
  assert.deepEqual([asItIs.size, asItIs.schema.floats, asItIs.schema.vector], [4, ['rating'], 3])
  // A fork holds the same, and is apart from then on.
  const fork = asItIs.fork('sandbox')
  assert.deepEqual([fork.size, fork.name, fork.schema.vector], [4, 'sandbox', 3])
  assert.deepEqual(keys(fork.query({ text: 'lease' })), keys(asItIs.query({ text: 'lease' })))
  fork.delete(1)
  fork.upsert({ key: 9, text: { title: 'new in the fork' } })
  assert.deepEqual([fork.size, asItIs.size, asItIs.get(9)], [4, 4, null])
  // What passes a filter, deleted at once: in the fork only.
  assert.equal(fork.deleteWhere(attr('kind').eq('pdf')), 1)
  assert.equal(fork.deleteWhere(attr('kind').eq('pdf')), 0)
  assert.deepEqual([fork.count(attr('kind').eq('pdf')), asItIs.count(attr('kind').eq('pdf'))], [0, 2])
  throws(() => fork.deleteWhere(), 'ERR_HINNY_INVALID', /filter/)
  assert.ok(db.namespaces().includes('sandbox'))
  throws(() => asItIs.fork('sandbox'), 'ERR_HINNY_INVALID', /already/)
  throws(() => asItIs.fork('not a name'), 'ERR_HINNY_INVALID', /letters/)
  fork.close()
  db.delete('sandbox')
  asItIs.close()
  assert.throws(() => db.open('nothing'), /no namespace/)
})

test('patch: part of a document changes, the rest stays', () => {
  // One attribute changes; the text, the other attributes and the vector stay.
  assert.equal(ns.patch(1, { attrs: { kind: 'scan' } }), true)
  assert.deepEqual(ns.get(1), { year: 2025, kind: 'scan' })
  assert.deepEqual(ns.text(1), { title: 'apartment lease', body: 'the lease for the apartment' })
  assert.deepEqual(keys(ns.query({ vector: [0, 0, 1], limit: 1, filter: attr('kind').eq('scan') })), [1])
  assert.deepEqual(keys(ns.query({ text: 'apartment', filter: attr('kind').eq('scan') })), [1])
  // Text, a vector (a Float32Array) and a version; `remove` takes an attribute and a text field away.
  assert.equal(ns.patch(1n, { text: { body: 'signed in May' }, vector: new Float32Array([1, 0, 0]), version: 5 }), true)
  assert.equal(ns.versions().get(1n), 5n)
  assert.deepEqual(keys(ns.query({ text: 'signed' })), [1])
  assert.equal(ns.patch(1, { remove: ['year', 'body'] }), true)
  assert.deepEqual(ns.get(1), { kind: 'scan' })
  assert.deepEqual(ns.text(1), { title: 'apartment lease' })
  assert.equal(ns.versions().get(1n), 5n)
  // A key that isn't there: false, nothing written.
  assert.equal(ns.patch(99, { attrs: { year: 1 } }), false)
  assert.equal(ns.get(99), null)
  // With a condition: written when it holds, not when it doesn't.
  assert.equal(ns.patch(2, { attrs: { year: 1999 } }, attr('kind').eq('pdf')), false)
  assert.equal(ns.get(2).year, 2024)
  assert.equal(ns.patch(2, { attrs: { year: 1999 } }, attr('kind').eq('jpg')), true)
  assert.equal(ns.get(2).year, 1999)
  assert.equal(ns.patch(2, { attrs: { year: 2000 } }, null), true, 'no condition')
  // Mistakes.
  throws(() => ns.patch(2, { attrs: { month: 3 } }), 'ERR_HINNY_INVALID', /month/)
  throws(() => ns.patch(2, { remove: ['nothing'] }), 'ERR_HINNY_INVALID', /nothing/)
  throws(() => ns.patch(2, { vector: [1, 2] }), 'ERR_HINNY_INVALID', /2 dimensions/)
  throws(() => ns.patch(2, { txt: {} }), 'ERR_HINNY_INVALID', /txt/)
  throws(() => ns.patch(2, { text: 'a string' }), 'ERR_HINNY_INVALID')
  throws(() => ns.patch(2, { remove: 'year' }), 'ERR_HINNY_INVALID')
  throws(() => ns.patch(2, 5), 'ERR_HINNY_INVALID')
  throws(() => ns.patch(2, {}, { year: 1 }), 'ERR_HINNY_INVALID', /attr\(/)
  throws(() => ns.patch(-1, {}), 'ERR_HINNY_INVALID')
  assert.equal(ns.get(2).year, 2000)
})

test('patchWhere: a patch for every document that passes a filter', () => {
  assert.equal(ns.patchWhere(attr('kind').eq('pdf'), { attrs: { kind: 'doc', year: 2000 } }), 2)
  assert.equal(ns.patchWhere(attr('kind').eq('pdf'), { attrs: { year: 1 } }), 0)
  assert.deepEqual([ns.count(attr('kind').eq('doc')), ns.count(attr('year').eq(2000)), ns.size], [2, 2, 4])
  assert.equal(ns.text(1).title, 'apartment lease')
  assert.equal(ns.patchWhere(attr('year').gte(2023), { remove: ['kind'] }), 2)
  assert.deepEqual([ns.count(attr('kind').exists()), ns.count(attr('kind').eq('doc'))], [2, 2])
  throws(() => ns.patchWhere(undefined, { attrs: { year: 1 } }), 'ERR_HINNY_INVALID', /filter/)
  throws(() => ns.patchWhere(attr('year').gt(0), { attrs: { month: 1 } }), 'ERR_HINNY_INVALID', /month/)
})

test('writes with a condition: upsertIf and deleteIf', () => {
  const newer = (rev, extra = {}) => ({ key: 10, text: { title: `revision ${rev}` }, attrs: { year: rev }, vector: [1, 1, 0], ...extra })
  // Written where there is none; refused when the condition fails; written when it holds.
  assert.equal(ns.upsertIf(newer(5), attr('year').lt(5)), true)
  assert.equal(ns.upsertIf(newer(4), attr('year').lt(5 - 1)), false)
  assert.deepEqual(ns.get(10), { year: 5 })
  assert.equal(ns.text(10).title, 'revision 5')
  assert.equal(ns.upsertIf(newer(6, { version: 3 }), attr('year').lt(6)), true)
  assert.deepEqual([ns.get(10), ns.versions().get(10n), ns.size], [{ year: 6 }, 3n, 5])
  // deleteIf both ways; a key that isn't there is false.
  assert.equal(ns.deleteIf(10, attr('year').eq(5)), false)
  assert.notEqual(ns.get(10), null)
  assert.equal(ns.deleteIf(10, attr('year').eq(6)), true)
  assert.equal(ns.get(10), null)
  assert.equal(ns.deleteIf(10n, attr('year').eq(6)), false)
  throws(() => ns.deleteIf(1), 'ERR_HINNY_INVALID', /attr\(/)
  throws(() => ns.upsertIf(newer(7)), 'ERR_HINNY_INVALID', /attr\(/)
  throws(() => ns.upsertIf({ key: 11, attrs: { month: 1 } }, attr('year').gt(0)), 'ERR_HINNY_INVALID', /month/)
})

test('typing: a word still being typed matches the words that start with it', () => {
  assert.deepEqual(keys(ns.query({ text: 'lea' })), [])
  assert.deepEqual(keys(ns.query({ text: 'lea', typing: 32 })).sort(), [1, 3])
  assert.deepEqual(keys(ns.query({ text: 'lea', typing: 0 })), [])
  assert.deepEqual(keys(ns.query({ text: 'the apart', typing: 32, allWords: true })), [1])
  assert.deepEqual(keys(ns.query({ text: 'lease ren', typing: 32, allWords: true })), [3])
  assert.deepEqual(keys(ns.explain({ text: 'renew', typing: 32 }).hits), [3])
  const snap = ns.snapshot()
  assert.deepEqual(keys(snap.query({ text: 'lea' })), [])
  assert.deepEqual(keys(snap.query({ text: 'lea', typing: 32 })).sort(), [1, 3])
  throws(() => ns.query({ typing: -1 }), 'ERR_HINNY_INVALID')
  throws(() => ns.query({ typing: 'yes' }), 'ERR_HINNY_INVALID')
})

test('deleting a namespace', () => {
  const temp = db.namespace('temp', { text: 'text' })
  temp.upsert({ key: 1, text: 'short lived' })
  const snap = temp.snapshot()
  // Refused while it is open; still whole afterwards.
  throws(() => db.delete('temp'), 'ERR_HINNY_LOCKED')
  assert.equal(temp.size, 1)
  assert.deepEqual(db.namespaces(), ['docs', 'temp'])
  // Closing the namespace closes its snapshot too.
  temp.close()
  throws(() => snap.count(), 'ERR_HINNY_CLOSED')
  db.delete('temp')
  assert.deepEqual(db.namespaces(), ['docs'])
  assert.equal(fs.existsSync(path.join(dir, 'temp')), false)
  assert.throws(() => db.open('temp'), /no namespace/)
  assert.throws(() => db.delete('temp'), /no namespace/)
  throws(() => db.delete('../docs'), 'ERR_HINNY_INVALID', /letters/)
  throws(() => db.delete(7), 'ERR_HINNY_INVALID')
  assert.equal(ns.size, 4)
  // Made again, it starts empty.
  assert.equal(db.namespace('temp', { text: 'text' }).size, 0)
  // A fork is deleted apart from the original.
  const fork = ns.fork('sandbox')
  throws(() => db.delete('sandbox'), 'ERR_HINNY_LOCKED')
  fork.close()
  db.delete('sandbox')
  assert.deepEqual([ns.size, db.namespaces()], [4, ['docs', 'temp']])
  // A closed database deletes nothing.
  const other = new Database(path.join(dir, 'other'))
  other.close()
  throws(() => other.delete('docs'), 'ERR_HINNY_CLOSED')
})

test('a namespace open in another process', () => {
  const code = `new (require(${JSON.stringify(path.join(__dirname, '..'))}).Database)(${JSON.stringify(dir)}).namespace('docs', ${JSON.stringify(SCHEMA)})`
  const r = spawnSync(process.execPath, ['-e', code], { encoding: 'utf8' })
  assert.notEqual(r.status, 0)
  assert.match(r.stderr, /ERR_HINNY_LOCKED/)
  assert.match(r.stderr, /already open for writing/)
})

test('synced writes survive a killed process', () => {
  const where = path.join(dir, 'killed')
  const lib = JSON.stringify(path.join(__dirname, '..'))
  // SIGKILL runs no finalizer: recovery must come from the synced log, not a final flush.
  const code = `
    const ns = new (require(${lib}).Database)(${JSON.stringify(where)}).namespace('docs', { text: 'body' })
    ns.upsertMany([{ key: 1, text: 'first' }, { key: 2, text: 'second' }])
    ns.delete(1)
    ns.sync()
    if (ns.stats().segments !== 0) throw new Error('flushed')
    process.kill(process.pid, 'SIGKILL')`
  const r = spawnSync(process.execPath, ['-e', code], { encoding: 'utf8' })
  assert.equal(r.signal, 'SIGKILL', r.stderr)
  const reopened = new Database(where).namespace('docs', { text: 'body' })
  assert.deepEqual([...reopened.versions().keys()], [2n])
  assert.deepEqual(reopened.text(2), { body: 'second' })
  reopened.close()
})

test('errors', () => {
  throws(() => ns.upsert({ key: 9, vector: [1, 2] }), 'ERR_HINNY_INVALID', /2 dimensions/)
  throws(() => ns.upsert({ key: 9, vector: new Float32Array(4) }), 'ERR_HINNY_INVALID', /4 dimensions/)
  throws(() => ns.query({ vector: [1, 0] }), 'ERR_HINNY_INVALID')
  throws(() => ns.upsert({ key: 9, text: { nope: 'x' } }), 'ERR_HINNY_INVALID', /nope/)
  throws(() => ns.upsert({ key: 9, attrs: { nope: 1 } }), 'ERR_HINNY_INVALID', /nope/)
  throws(() => ns.upsert({ key: 9, attrs: { year: 2.5 } }), 'ERR_HINNY_INVALID', /another type/)
  throws(() => ns.upsert({ key: 9, attrs: { kind: 3 } }), 'ERR_HINNY_INVALID', /another type/)
  throws(() => ns.upsert({ key: 9, attrs: { year: [2025] } }), 'ERR_HINNY_INVALID')
  throws(() => ns.upsert({ key: 9, text: 'one field only' }), 'ERR_HINNY_INVALID', /one text field/)
  throws(() => ns.upsert({ key: 9, vector: new Int32Array(3) }), 'ERR_HINNY_INVALID', /Float32Array/)
  throws(() => ns.upsert({ key: 9, vector: [1, 'x', 0] }), 'ERR_HINNY_INVALID')
  throws(() => ns.upsert({ text: { title: 'no key' } }), 'ERR_HINNY_INVALID', /key/)
  throws(() => ns.upsert({ key: 9, txt: 'typo' }), 'ERR_HINNY_INVALID', /txt/)
  throws(() => ns.upsert(9), 'ERR_HINNY_INVALID')
  const sc = db.namespace('scored', { text: 'body', floats: ['score'] })
  throws(() => sc.upsert({ key: 1, text: 'x', attrs: { score: NaN } }), 'ERR_HINNY_INVALID', /NaN/)
  assert.equal(sc.get(1), null)
  throws(() => ns.query({ limt: 3 }), 'ERR_HINNY_INVALID', /limt/)
  throws(() => ns.query({ limit: -1 }), 'ERR_HINNY_INVALID')
  throws(() => ns.query({ limit: '3' }), 'ERR_HINNY_INVALID')
  throws(() => ns.query({ fusion: 'mean' }), 'ERR_HINNY_INVALID', /fusion/)
  throws(() => ns.query({ textWeight: 2 }), 'ERR_HINNY_INVALID')
  throws(() => ns.query({ filter: { year: 2025 } }), 'ERR_HINNY_INVALID', /attr\(/)
  throws(() => ns.count(attr('year')), 'ERR_HINNY_INVALID', /attr\(/)
  throws(() => attr('year').gte('2025'), 'ERR_HINNY_INVALID', /not a string/)
  throws(() => attr('year').eq({}), 'ERR_HINNY_INVALID')
  throws(() => db.namespace('bad name!', { text: 'body' }), 'ERR_HINNY_INVALID', /namespace names/)
  throws(() => db.namespace('x', { text: 'body', language: 'klingon' }), 'ERR_HINNY_INVALID', /klingon/)
  throws(() => db.namespace('x', { text: 'body', vectors: 3 }), 'ERR_HINNY_INVALID', /vectors/)
  throws(() => db.namespace('x', { text: { body: -1 } }), 'ERR_HINNY_INVALID', /boost/)
  throws(() => db.namespace('x', { text: 'body', ints: ['body'] }), 'ERR_HINNY_INVALID', /twice/)
  assert.throws(() => new hinny.Filter(), Error)
})

test('stored text, snippets and highlights in UTF-16', () => {
  const filler = 'Nothing about it here. '.repeat(40)
  const body = `${filler}Die Miete für März 😀 und die Wohnung in der Elm Street. ${filler}`
  ns.upsert({ key: 10, text: { title: 'Mietvertrag', body }, attrs: { year: 2026 } })
  const h = ns.query({ text: 'miete märz wohnung', limit: 1 })[0]
  assert.equal(h.key, 10n)
  assert.deepEqual(h.text, { title: 'Mietvertrag', body })
  assert.equal(h.snippetField, 'body')
  // Offsets index the JavaScript string (UTF-16), so text after an emoji slices correctly.
  assert.deepEqual(h.highlights.map(([a, b]) => h.snippet.slice(a, b)), ['Miete', 'März', 'Wohnung'])
  assert.ok(h.snippet.length <= 250)
  assert.equal(ns.text(10).title, 'Mietvertrag')
  const lean = ns.query({ text: 'miete', snippet: 0, withText: false, limit: 1 })[0]
  assert.deepEqual([lean.text, lean.snippet, lean.snippetField, lean.highlights], [{}, null, null, []])
  const bare = db.namespace('bare', { text: 'body', storeText: false })
  bare.upsert({ key: 1, text: 'the lease' })
  const b = bare.query({ text: 'lease' })[0]
  assert.deepEqual([b.text, b.snippet], [{}, null])
  assert.deepEqual(bare.text(1), {}) // the document is there; no text was kept
})

test('text that isnt Unicode', () => {
  // Lone surrogates (from JSON escapes, in real mail) are replaced, not refused.
  const u = db.namespace('u', { text: 'body' })
  u.upsert({ key: 1, text: 'bad \ud83d surrogate lease' })
  u.upsertMany([{ key: 2, text: { body: 'also \udc00 lease' } }])
  assert.deepEqual(keys(u.query({ text: 'lease' })).sort(), [1, 2])
  assert.ok(u.text(1).body.includes('�'))
})

test('fusion, explain and candidate budgets', () => {
  const q = { text: 'lease', vector: [1, 0, 0], limit: 4 }
  assert.equal(ns.query({ ...q, textWeight: 1 })[0].key, ns.query({ text: 'lease', limit: 1 })[0].key)
  assert.equal(ns.query({ ...q, textWeight: 0 })[0].key, ns.query({ vector: [1, 0, 0], limit: 1 })[0].key)
  assert.equal(ns.query({ ...q, fusion: 'rrf' }).length, 4)

  const r = ns.explain({ text: 'lease', vector: [0, 0, 1], limit: 3 })
  assert.deepEqual(keys(r.hits), keys(ns.query({ text: 'lease', vector: [0, 0, 1], limit: 3 })))
  const f = r.fusion
  assert.deepEqual([f.lexical.depth, f.vector.depth], [200, 100])
  assert.equal(f.fused, f.lexical.found + f.vector.found - f.both)
  const [lo, hi] = f.lexical.scores
  assert.ok(lo <= hi)
  const budgets = ns.explain({ text: 'lease', vector: [0, 0, 1], limit: 1, textCandidates: 2, vectorCandidates: 3 }).fusion
  assert.deepEqual([budgets.lexical.depth, budgets.vector.depth], [2, 3])
  assert.equal(ns.explain({ text: 'lease' }).fusion, null)
  assert.equal(ns.snapshot().explain({ vector: [0, 0, 1] }).fusion, null)
})

test('limits and timeouts', () => {
  throws(() => ns.query({ text: 'lease', vector: [0, 0, 1], timeout: 0 }), 'ERR_HINNY_TIMEOUT')
  assert.equal(ns.query({ text: 'lease', timeout: 60_000 }).length, 2)
  throws(() => ns.query({ text: 'lease', timeout: -1 }), 'ERR_HINNY_INVALID')
  throws(() => ns.query({ text: Array.from({ length: 5000 }, (_, i) => `w${i}`).join(' ') }), 'ERR_HINNY_INVALID')
  ns.setLimits({ maxBatchBytes: 200 })
  throws(() => ns.upsert({ key: 9, text: { title: 'x'.repeat(500) } }), 'ERR_HINNY_INVALID')
  assert.ok(!keys(ns.query({ text: 'x'.repeat(10) })).includes(9))
  ns.setLimits({ maxBatchBytes: 1 << 30, flushBytes: 256 << 20 })
  ns.upsert({ key: 9, text: { title: 'x'.repeat(500) } })
  throws(() => ns.setLimits({ maxBatch: 1 }), 'ERR_HINNY_INVALID')
  const s = ns.stats()
  assert.ok(s.memtableBytes > 0)
  assert.equal(s.pinnedSegments, 0)
  assert.equal(s.documents, 5)
})

test('language', () => {
  const en = db.namespace('en', { text: 'body', language: 'english' })
  en.upsert({ key: 1, text: 'The tax returns were filed in March' })
  en.upsert({ key: 2, text: 'Running shoes for runners' })
  assert.deepEqual(keys(en.query({ text: 'taxed filings' })), [1])
  const h = en.query({ text: 'taxing', limit: 1 })[0]
  assert.deepEqual(h.highlights.map(([a, b]) => h.snippet.slice(a, b)), ['tax'])
  assert.deepEqual(keys(en.query({ text: 'the of and' })), [])
  en.close()
  throws(() => db.namespace('en', { text: 'body' }), 'ERR_HINNY_INVALID') // fixed once there's data
  const kept = db.namespace('kept', { text: 'body', language: 'english', stopwords: false })
  kept.upsert({ key: 1, text: 'the who' })
  assert.deepEqual(keys(kept.query({ text: 'the' })), [1])
})

test('sparse fields', () => {
  const sp = db.namespace('sp', { text: 'body', sparse: ['terms'], vector: 2 })
  assert.deepEqual(sp.schema.sparse, ['terms'])
  assert.deepEqual(sp.schema.text, { body: 1 })
  sp.upsertMany([
    { key: 1, text: 'a lease', sparse: { terms: { lease: 1.4, rent: 0.6 } }, vector: [1, 0] },
    { key: 2, text: 'a rent', sparse: { terms: [['rent', 1.0]] }, vector: [0, 1] },
  ])
  sp.upsert({ key: 3, text: 'nothing', sparse: { terms: { beach: 2 } }, vector: [0.7, 0.7] })
  // Dot product of query and document weights: 1 → 2·1.4 + 1·0.6 = 3.4, 2 → 1·1.0.
  const hits = sp.query({ sparse: { terms: { lease: 2, rent: 1 } } })
  assert.deepEqual(keys(hits), [1, 2])
  assert.ok(Math.abs(hits[0].textScore - 3.4) < 0.05)
  assert.ok(Math.abs(hits[1].textScore - 1.0) < 0.05)
  sp.flush()
  assert.deepEqual(keys(sp.query({ sparse: { terms: { beach: 1 } } })), [3])
  assert.deepEqual(keys(sp.query({ sparse: { terms: { rent: 1 } }, vector: [0, 1], limit: 1 })), [2])
  assert.deepEqual(keys(sp.query({ sparse: { terms: { rent: 1 } }, vector: [0, 1], limit: 1, sparseCandidates: 5 })), [2])
  // Sparse terms aren't words: text queries don't see them.
  assert.deepEqual(keys(sp.query({ text: 'beach' })), [])
  throws(() => sp.query({ sparse: { body: { lease: 1 } } }), 'ERR_HINNY_INVALID')
  throws(() => sp.upsert({ key: 4, sparse: { terms: [['lease']] } }), 'ERR_HINNY_INVALID')
})

test('maintenance: sync, flush, optimize, verify, stats', () => {
  ns.sync()
  for (let i = 0; i < 3; i++) {
    ns.upsert({ key: 100 + i, text: { title: `Extra ${i}` }, attrs: { year: 2020 }, vector: [0, 0, 1] })
    ns.flush()
  }
  const q = { text: 'lease', vector: [1, 0, 0], limit: 5 }
  const before = keys(ns.query(q))
  assert.ok(ns.stats().segments > 1)
  ns.optimize()
  const s = ns.stats()
  assert.equal(s.segments, 1)
  assert.equal(s.documents, 7)
  assert.equal(typeof s.merging, 'boolean')
  assert.deepEqual(keys(ns.query(q)), before)
  const v = ns.verify()
  assert.equal(v.segments, 1)
  assert.equal(v.uncheckedSegments, 0)
  assert.ok(v.textFiles >= 1)
})

test('async maintenance runs off the event loop', async () => {
  for (let i = 0; i < 3; i++) {
    ns.upsert({ key: 100 + i, text: { title: `Extra ${i}` }, vector: [0, 0, 1] })
    await ns.flushAsync()
  }
  const p = ns.optimizeAsync()
  assert.ok(p instanceof Promise)
  // Queries and writes run while it does.
  assert.equal(ns.query({ text: 'lease' }).length, 2)
  ns.upsert({ key: 200, text: { title: 'written during the merge' } })
  assert.equal(await p, undefined)
  assert.notEqual(ns.get(200), null)
  ns.delete(200)
  await ns.optimizeAsync()
  assert.equal(ns.stats().segments, 1)
  assert.deepEqual(await ns.verifyAsync(), { segments: 1, uncheckedSegments: 0, textFiles: ns.verify().textFiles })
  ns.close()
  throws(() => ns.optimizeAsync(), 'ERR_HINNY_CLOSED')
})

test('async errors carry a code', async () => {
  // Damage a segment's file, then verify.
  ns.flush()
  const nsDir = path.join(dir, 'docs')
  const seg = path.join(nsDir, fs.readdirSync(nsDir).find((f) => f.startsWith('seg-')))
  const bytes = fs.readFileSync(seg)
  bytes[bytes.length >> 1] ^= 0xff
  fs.writeFileSync(seg, bytes)
  await assert.rejects(ns.verifyAsync(), (e) => e instanceof Error && e.code === 'ERR_HINNY_IO')
  throws(() => ns.verify(), 'ERR_HINNY_IO')
})

test('vector metrics', () => {
  // Against the query [1, 0]: by cosine the short [0.5, 0] in the query's direction comes first
  // and [3, 1] second; by Euclidean distance [1, 1] is nearer than [3, 1]; by dot product the
  // long [3, 1] comes first and the short one last.
  const vectors = [{ key: 1, vector: [0.5, 0] }, { key: 2, vector: [3, 1] }, { key: 3, vector: [1, 1] }]
  const cases = {
    cosine: [[1, 1], [2, 3 / Math.sqrt(10)], [3, 1 / Math.sqrt(2)]],
    euclidean: [[1, -0.25], [3, -1], [2, -5]], // minus the squared distance
    dot: [[2, 3], [3, 1], [1, 0.5]], // the product
  }
  for (const [metric, want] of Object.entries(cases)) {
    const m = db.namespace(metric, { vector: 2, metric })
    assert.equal(m.schema.vector, 2)
    assert.equal(m.schema.metric, metric)
    m.upsertMany(vectors)
    const hits = m.query({ vector: [1, 0] })
    assert.deepEqual(keys(hits), want.map(([k]) => k), metric)
    hits.forEach((h, i) => assert.ok(Math.abs(h.score - want[i][1]) < 1e-4, `${metric}: ${h.key} scores ${h.score}`))
    m.close()
    const reopened = db.open(metric)
    assert.equal(reopened.schema.metric, metric)
    reopened.close()
    const other = metric === 'dot' ? 'euclidean' : 'dot'
    throws(() => db.namespace(metric, { vector: 2, metric: other }), 'ERR_HINNY_INVALID', /another schema/)
  }
  assert.equal(db.namespace('plain', { vector: 2 }).schema.metric, 'cosine')
  assert.equal(db.namespace('novector', { text: 'body' }).schema.metric, null)
  throws(() => db.namespace('other', { vector: 2, metric: 'manhattan' }), 'ERR_HINNY_INVALID', /cosine, euclidean or dot/)
  throws(() => db.namespace('other', { text: 'body', metric: 'dot' }), 'ERR_HINNY_INVALID', /vector/)
})

test('a patch gives sparse terms again', () => {
  const sp = db.namespace('sp', { text: 'body', sparse: ['terms'], ints: ['year'] })
  sp.upsert({ key: 1, text: 'a lease', sparse: { terms: { lease: 1.4 } }, attrs: { year: 2025 } })
  const found = () => keys(sp.query({ sparse: { terms: { lease: 1 } } }))
  assert.deepEqual(found(), [1])
  // The terms aren't kept as given: a patch that leaves them out is refused.
  throws(() => sp.patch(1, { attrs: { year: 2026 } }), 'ERR_HINNY_INVALID', /sparse/)
  throws(() => sp.patchWhere(attr('year').eq(2025), { attrs: { year: 2026 } }), 'ERR_HINNY_INVALID', /sparse/)
  assert.deepEqual(sp.get(1), { year: 2025 })
  assert.equal(sp.patch(1, { attrs: { year: 2026 }, sparse: { terms: { lease: 1.4 } } }), true)
  assert.deepEqual([sp.get(1), found()], [{ year: 2026 }, [1]])
  assert.equal(sp.patchWhere(attr('year').eq(2026), { attrs: { year: 2027 }, sparse: { terms: [['lease', 1.4]] } }), 1)
  assert.deepEqual([sp.get(1), found()], [{ year: 2027 }, [1]])
  // Or that removes the field.
  assert.equal(sp.patch(1, { attrs: { year: 2028 }, remove: ['terms'] }), true)
  assert.deepEqual([sp.get(1), found()], [{ year: 2028 }, []])
  throws(() => sp.patch(1, { sparse: ['lease'] }), 'ERR_HINNY_INVALID')
})

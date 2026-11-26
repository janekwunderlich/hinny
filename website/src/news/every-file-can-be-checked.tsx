import * as stylex from '@stylexjs/stylex'
import { InlineCode } from '#/components/CodeBlock'
import { Figure } from '#/components/Figure'
import { Link } from '#/components/Link'
import { H2, Item, List, P } from '#/components/Prose'
import { DataTable } from '#/components/Table'
import { focus, link } from '#/components/styles'
import type { Post } from './index'
import { Blocks, Cover, InDevelopment } from './parts'

/* Every statement here is from DESIGN.md ("Formats", "Durability and recovery contract") or the
   dev log's entries on reliability. If those change, change them here too. */

function Body() {
  return (
    <>
      <InDevelopment />
      <P>
        A search index is derived from other data, and it is tempting to treat it as disposable. But rebuilding one means reading and embedding everything again, and an index that quietly returns wrong results is worse than one that is gone. Two rules run through Hinny's storage: a damaged file must be noticed, and a file that cannot be understood must be refused, never guessed at.
      </P>

      <H2 id="formats">Every file says what it is</H2>
      <P>
        Every file starts by naming its kind and the version of its format: the manifest, each write-ahead log, each deletion file, each segment, each file of stored text, and the schema. A release reads the formats of the release before it and writes its own; older segments are rewritten when a merge reaches them. A file of a newer format is refused with an error that names both versions, never misread.
      </P>
      <P>The repository keeps a database written by each release, and a test checks that the current code reads it, writes on top of it and merges it.</P>

      <H2 id="checksums">A checksum on every file</H2>
      <Figure wide caption="The files of a namespace and what guards each. Segment checksums are in segment format 8 and later; segments from before it are rewritten, with checksums, when a merge reaches them.">
        <DataTable
          wide
          words
          label="The files of a namespace, their checksums, and when each is checked"
          head={['File', 'What it holds', 'Checksum', 'Checked']}
          rows={[
            ['Manifest', 'Which segments make up the index', 'CRC32 of the whole file', 'When the index is opened'],
            ['Write-ahead log', 'Writes that are not in a segment yet', 'CRC32 of every record; a batch is one record', 'When the log is replayed'],
            ['Deletion file', "Which of a segment's documents are deleted", 'A header and a CRC32', 'When the index is opened'],
            ['Segment', 'Posting lists, vector codes, attribute columns', 'CRC32 of the header, and one of every mebibyte after it', 'The header when the segment is opened; the rest by verify'],
            ['Stored text', "The documents' text, compressed in blocks", "zstd's checksum in every block", 'When a block is read'],
          ]}
        />
      </Figure>
      <P>
        A batch of writes is one log record under one checksum, so after a crash a batch is there whole or not at all, and a record torn off at the end of the log is ignored on replay. A manifest or a deletion file that fails its checksum is refused: before format 4, damage that still parsed was believed.
      </P>
      <Figure boxed caption="A segment file from format 8 on. The header's checksum is checked whenever the segment is opened; the checksums of the body, one for each mebibyte, are at the end of the file.">
        <Blocks
          label="A segment file: a header with its checksum, the body in mebibytes, then one checksum for each mebibyte"
          blocks={[
            { name: 'Header', note: 'what, which version' },
            { name: 'CRC32', note: 'of the header', kind: 'marked' },
            { name: '1 MiB', note: 'of the body' },
            { name: '1 MiB', note: 'of the body' },
            { name: '1 MiB', note: 'of the body' },
            { name: '…', kind: 'faint' },
            { name: 'CRC32 × n', note: 'one per MiB', kind: 'marked' },
          ]}
        />
      </Figure>

      <H2 id="verify">Verify reads everything</H2>
      <P>
        Segments are memory-mapped, not read through a call that could check them, so a query does not verify checksums as it goes: that would mean reading whole files to answer from a small part of them. Instead, <InlineCode>verify</InlineCode> on a namespace (<InlineCode>ns.verify()</InlineCode> in Python, <InlineCode>hinny_verify</InlineCode> in C) reads every file against its checksums and names the first damaged one. Reading everything is what it costs. Segments from before format 8 have no checksums, and verify counts them as unchecked.
      </P>

      <H2 id="refuse">Refusing instead of guessing</H2>
      <P>
        The dangerous moment for a damaged index is opening it. An audit of the recovery code found that an unreadable manifest opened as an empty index, whose cleanup then deleted the segments it no longer named. Now opening rejects an unreadable or malformed manifest or deletion file before anything is cleaned up. A missing manifest beside existing segment files requires explicit recovery, and the files are kept. Every log's header is checked before a writer is created.
      </P>
      <P>
        Failures while writing are handled in the same spirit. A failed append to the log stops the writer until the index is reopened, because appending behind a torn record could lose later records on replay. A deletion bitmap whose save failed stays marked as unsaved. A flush or merge that fails to publish keeps the previous state and its files, to be retried.
      </P>

      <H2 id="tests">Tested by breaking it</H2>
      <Figure caption="The tests are in the repository's crates/hinny/tests. The fault sweep was 296 cases before torn log appends were added to it.">
        <DataTable
          words
          label="Tests that damage or interrupt the index on purpose"
          head={['Test', 'What it breaks', 'What must hold']}
          rows={[
            ['integrity', "Single bits in a segment's postings, its header, its checksums and in stored text", 'Verify finds every one'],
            ['faults', 'Each of the engine\'s writes and syncs, failed in turn at every point a workload reaches it: 344 cases, including log appends torn in half', 'The index reopens to exactly the state after a prefix of the batches, with every batch acknowledged before the last successful sync'],
            ['kills', 'A writing process, killed with SIGKILL at a random moment while flushes and merges run, 40 times', 'Every acknowledged batch survives, whole and in order'],
            ['formats', 'Nothing: it opens a database written by each earlier release', 'The current code reads it, writes on top of it and merges it'],
          ]}
        />
      </Figure>
      <P>
        The tests are checked against the bugs they exist for. Putting back a deletion bug that had just been fixed makes the fault sweep fail, and removing the flush of the log on each write makes the kill test lose acknowledged batches at once.
      </P>

      <H2 id="limits">What it does not promise</H2>
      <List>
        <Item>
          A write is acknowledged once its WAL record reaches the operating system. That survives a process crash; an OS crash or power loss can lose writes made since the last successful <InlineCode>sync()</InlineCode> or <InlineCode>flush()</InlineCode>.
        </Item>
        <Item>Synchronization relies on the filesystem and storage device honoring sync requests. On non-Unix platforms, directory entries are not synchronized.</Item>
        <Item>The crash tests kill processes and simulate torn log tails; they are not a hardware power-loss certification.</Item>
        <Item>From format 7 onward, deletion files are numbered and named in the manifest, so removing one externally causes open to fail. For legacy pre-format-7 segments, a missing deletion file still means no deletions.</Item>
        <Item>Checksums are verified on file open, when reading stored text blocks, and during <InlineCode>verify()</InlineCode>. They are not checked on every query.</Item>
      </List>
      <P>
        How writes, segments and merges fit together is in the{' '}
        <Link to="/docs" hash="how" {...stylex.props(link.inline, focus.ring)}>
          docs
        </Link>
        , and the dated history of this work, commit by commit, is in the{' '}
        <Link to="/log" {...stylex.props(link.inline, focus.ring)}>
          dev log
        </Link>
        .
      </P>
    </>
  )
}

export const storage: Post = {
  slug: 'every-file-can-be-checked',
  title: 'Every file can be checked',
  short: 'Every file can be checked',
  date: '2026-11-13',
  category: 'Engineering',
  coat: 'clover',
  summary: "What Hinny's storage format does about damaged files: a checksum on every file, a version on every format, an index that refuses to open rather than guess, and tests that break it on purpose.",
  lede: 'An embedded database keeps its data in files beside the application, where disks fail and processes are killed in the middle of a write. This is what Hinny does so that damage is found, not believed.',
  Cover: () => (
    <Cover
      coat="clover"
      lines={[
        ['manifest', 'CRC32'],
        ['write-ahead log', 'CRC32 per record'],
        ['deletion file', 'CRC32'],
        ['segment', 'CRC32 per MiB'],
        ['stored text', 'zstd checksum per block'],
      ]}
    />
  ),
  Body,
}

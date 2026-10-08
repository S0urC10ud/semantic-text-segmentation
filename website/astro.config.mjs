import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Set TYPESEG_BASE=/semantic-text-segmentation for a repository Pages preview.
const base = process.env.TYPESEG_BASE || '/';
export default defineConfig({
  site: 'https://typeseg.martin-dallinger.me',
  base,
  output: 'static',
  trailingSlash: 'always',
  integrations: [starlight({
    title: 'TypeSeg',
    description: 'See the content types inside a text file. Local segmentation for security analysis.',
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/S0urC10ud/semantic-text-segmentation' }],
    customCss: ['./src/styles/custom.css'],
    sidebar: [
      { label: 'Introduction', items: [
        { label: 'Overview', slug: 'introduction/overview' },
        { label: 'Web Demo', slug: 'introduction/web-demo' },
      ] },
      { label: 'Getting Started', items: [
        { label: 'Installation', slug: 'getting-started/installation' },
        { label: 'Quick Start', slug: 'getting-started/quick-start' },
      ] },
      { label: 'Core Concepts', items: [
        { label: 'How TypeSeg Works', slug: 'core-concepts/how-typeseg-works' },
        { label: 'Models & Content Types', slug: 'core-concepts/models-and-content-types' },
        { label: 'Understanding the Output', slug: 'core-concepts/understanding-the-output' },
      ] },
      { label: 'CLI & Bindings', items: [
        { label: 'Command Line', slug: 'cli/command-line' },
      ] },
      { label: 'Security Workflows', items: [
        { label: 'For Analysis Platforms', slug: 'security/analysis-platforms' },
        { label: 'Performance & Evaluation', slug: 'security/performance' },
      ] },
      { label: 'Resources', items: [
        { label: 'Known Limitations', slug: 'resources/limitations' },
        { label: 'Research & Citation', slug: 'resources/research' },
      ] },
    ],
  })],
  redirects: { '/': `${base.replace(/\/$/, '')}/introduction/overview/` },
});

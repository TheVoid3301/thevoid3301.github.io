// @ts-check

import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import starlightBlog from 'starlight-blog';

// Astro 7 Markdown Processor
import { unified } from '@astrojs/markdown-remark';

// LaTeX / KaTeX
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	site: 'https://thevoid3301.github.io',
	markdown: {
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
  	},
	integrations: [
		mermaid({
			autoTheme: true,
			enableLog: false,
		}),

		starlight({
			customCss: [
				'./src/styles/geek.css',
				'katex/dist/katex.min.css',
			],

			title: "TheVoid3301's Blog",

			expressiveCode: {
				themes: ['github-dark', 'github-light'],
				useStarlightDarkModeSwitch: true,
				styleOverrides: {
					borderRadius: '0.75rem',
					codeFontSize: '0.9rem',
					codeLineHeight: '1.7',
					codeFontFamily:
					"'JetBrains Mono', 'Cascadia Code', Consolas, monospace",
				},
			},

			components: {
				PageTitle: './src/components/AriticleTitle.astro',
			},

			locales: {
				root: {
					label: '简体中文',
					lang: 'zh-CN',
				}
			},

			social: [
				{
					icon: 'github', 
				    label: 'GitHub', 
					href: 'https://github.com/thevoid3301' 
				},
			],

			plugins: [
				starlightBlog({
					title: '碎碎念',
					prefix: 'notes',
					rss: true,
					postCount: 10,
					recentPostCount: 5,
					metrics: {
						readingTime: true,
					},
				}),
			],

			sidebar: [
				{
					label: 'Algorithm',
					items: [{ autogenerate: { directory: 'algorithms' } }],
				},
				{
					label: 'Infra',
					items: [{ autogenerate: { directory: 'infra' } }],
				},
				{
					label: 'Project',
					items: [{ autogenerate: { directory: 'projects' } }],
				},
				{
					label: 'research',
					items: [{ autogenerate: { directory: 'research' } }],
				},
				{
					label: '文章归档',
					link: '/archive',
				}
			],
		}),
	],
});

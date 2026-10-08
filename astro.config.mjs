// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://thevoid3301.github.io',
	integrations: [
		starlight({
			customCss: ['./src/styles/geek.css'],
			title: "TheVoid3301's Blog",
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
			sidebar: [
				{
					label: 'Note',
					items: [{ autogenerate: { directory: 'notes' } }],
				},
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
			],
		}),
	],
});

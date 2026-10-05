#!/usr/bin/env node

/**
 * update-bypass-list.js
 *
 * Syncs a repo's PR interaction-limits bypass list with the current
 * set of eligible external contributors.
 *
 * Usage:
 *   GITHUB_TOKEN=xxxx node update-bypass-list.js <owner> <repo> [--dry-run]
 */

import { findEligibleContributors } from './find-eligible-contributors.js';

const GITHUB_API = 'https://api.github.com';

function authHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2026-03-10',
    'User-Agent': 'update-bypass-list-script',
    'Content-Type': 'application/json',
  };
}

function bypassListUrl(owner, repo) {
  return `${GITHUB_API}/repos/${owner}/${repo}/interaction-limits/pulls/bypass-list`;
}

async function fetchBypassList(owner, repo, token) {
  const url = bypassListUrl(owner, repo);
  const res = await fetch(url, { headers: authHeaders(token) });

  if (res.status === 404) {
    return new Set();
  }

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Failed to fetch bypass list (${res.status}): ${body}`);
  }

  const data = await res.json();
  const logins = data.map((user) => user?.login).filter(Boolean);

  return new Set(logins);
}

async function addUsersToBypassList(owner, repo, token, usernames) {
  if (usernames.length === 0) return;

  const res = await fetch(bypassListUrl(owner, repo), {
    method: 'PUT',
    headers: authHeaders(token),
    body: JSON.stringify({ users: usernames }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Failed to add users [${usernames.join(', ')}] (${res.status}): ${body}`);
  }
}

async function removeUsersFromBypassList(owner, repo, token, usernames) {
  if (usernames.length === 0) return;

  const res = await fetch(bypassListUrl(owner, repo), {
    method: 'DELETE',
    headers: authHeaders(token),
    body: JSON.stringify({ users: usernames }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Failed to remove users [${usernames.join(', ')}] (${res.status}): ${body}`);
  }
}

function diffLists(eligibleLogins, currentBypassLogins) {
  const toAdd = [...eligibleLogins].filter((login) => !currentBypassLogins.has(login));
  const toRemove = [...currentBypassLogins].filter((login) => !eligibleLogins.has(login));
  return { toAdd, toRemove };
}

/**
 * Update the bypass list to eligible contributors.
 *
 * @param {Object} params
 * @param {string} params.owner
 * @param {string} params.repo
 * @param {string} params.token
 * @param {boolean} [params.dryRun] - If true, only computes the diff, makes no API writes.
 */
async function updateBypassList({ owner, repo, token, dryRun = false }) {
  console.error(`Getting eligible external contributors for ${owner}/${repo}...`);
  const eligibleLogins = new Set(await findEligibleContributors({ owner, repo, token }));
  console.error(`Found ${eligibleLogins.size} eligible user(s).`);

  console.error(`Fetching current bypass list...`);
  const currentBypassLogins = await fetchBypassList(owner, repo, token);
  console.error(`Current bypass list has ${currentBypassLogins.size} user(s).`);

  const { toAdd, toRemove } = diffLists(eligibleLogins, currentBypassLogins);

  console.error(`\nUsers to add (${toAdd.length}): ${toAdd.join(', ') || '(none)'}`);
  console.error(`Users to remove (${toRemove.length}): ${toRemove.join(', ') || '(none)'}`);

  if (dryRun) {
    console.error('\nDry run mode: no changes will be made.');
    return;
  }

  try {
    await addUsersToBypassList(owner, repo, token, toAdd);
    if (toAdd.length > 0) console.error(`Added ${toAdd.length} user(s) to bypass list.`);
  } catch (err) {
    console.error(`Error adding users: ${err.message}`);
  }

  try {
    await removeUsersFromBypassList(owner, repo, token, toRemove);
    if (toRemove.length > 0) console.error(`Removed ${toRemove.length} user(s) from bypass list.`);
  } catch (err) {
    console.error(`Error removing users: ${err.message}`);
  }
}

// ---- CLI entry point ----

function isRunAsCLI() {
  return import.meta.url === `file://${process.argv[1]}`;
}

async function runCLI() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const [owner, repo] = args.filter((a) => !a.startsWith('--'));

  if (!owner || !repo) {
    console.error('Usage: node update-bypass-list.js <owner> <repo> [--dry-run]');
    process.exit(1);
  }

  const token = process.env.GITHUB_TOKEN;
  if (!token) {
    console.error('Error: set GITHUB_TOKEN environment variable with a valid GitHub token.');
    process.exit(1);
  }

  await updateBypassList({ owner, repo, token, dryRun });
}

if (isRunAsCLI()) {
  runCLI().catch((err) => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}

export { updateBypassList };

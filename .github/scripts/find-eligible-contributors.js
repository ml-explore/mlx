#!/usr/bin/env node

/**
 * find-eligible-contributors.js
 *
 * Finds users who have opened pull requests on a GitHub repo,
 * are NOT collaborators on that repo, and match:
 *   - fewer than 5 currently open PRs
 *   - more than 2 merged PRs
 *   - merge rate (merged / closed) > 50%
 *
 * Usage:
 *   GITHUB_TOKEN=xxxx node find-eligible-contributors.js <owner> <repo>
 */

const GITHUB_API = 'https://api.github.com';

function authHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'find-eligible-contributors-script',
  };
}

async function fetchAllPages(url, token, { stopWhen } = {}) {
  const results = [];
  let page = 1;
  const perPage = 100;
  const concurrentRequests = 5;

  while (true) {
    const responses = await Promise.all(
      Array.from({ length: concurrentRequests }, async (_, index) => {
        const pageUrl = new URL(url);
        pageUrl.searchParams.set('per_page', String(perPage));
        pageUrl.searchParams.set('page', String(page + index));

        const res = await fetch(pageUrl, { headers: authHeaders(token) });
        if (!res.ok) {
          const body = await res.text();
          throw new Error(`GitHub API error ${res.status} for ${pageUrl}: ${body}`);
        }
        return res.json();
      })
    );

    let end = false;
    for (const data of responses) {
      if (!Array.isArray(data) || data.length === 0) {
        end = true;
        continue;
      }
      if (data.length < perPage) {
        end = true;
      }

      for (const item of data) {
        if (stopWhen && stopWhen(item)) break;
        results.push(item);
      }
    }

    if (end) {
      break;
    }

    page += concurrentRequests;
  }

  return results;
}

async function fetchPullRequests(owner, repo, token) {
  const url = `${GITHUB_API}/repos/${owner}/${repo}/pulls?state=all&sort=created&direction=desc`;

  const oneYearAgo = new Date();
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);

  return fetchAllPages(url, token, {
    stopWhen: (pr) => new Date(pr.created_at) < oneYearAgo,
  });
}

async function fetchCollaborators(owner, repo, token) {
  const url = `${GITHUB_API}/repos/${owner}/${repo}/collaborators`;
  const collaborators = await fetchAllPages(url, token);
  return new Set(collaborators.map((c) => c.login));
}

/**
 * Aggregate per-user PR stats from the list of PRs.
 * Returns Map<login, {
 *   open: number,
 *   closed: number,
 *   merged: number,
 *   mostRecentCreatedAt: Date
 * }>
 */
function aggregateUserStats(pullRequests) {
  const stats = new Map();

  for (const pr of pullRequests) {
    const login = pr.user?.login;
    if (!login) continue;

    if (!stats.has(login)) {
      stats.set(login, {
        open: 0,
        closed: 0,
        merged: 0,
        mostRecentCreatedAt: null,
      });
    }
    const s = stats.get(login);

    if (pr.state === 'open') {
      s.open += 1;
    } else if (pr.state === 'closed') {
      s.closed += 1;
      if (pr.merged_at) {
        s.merged += 1;
      }
    }

    const createdAt = new Date(pr.created_at);
    if (!s.mostRecentCreatedAt || createdAt > s.mostRecentCreatedAt) {
      s.mostRecentCreatedAt = createdAt;
    }
  }

  return stats;
}

/**
 * Apply eligibility filters:
 *   - fewer than 5 currently open PRs
 *   - more than 2 merged PRs
 *   - merge rate (merged / closed) > 50%
 *   - not a collaborator
 *   - has an open PR or at least one PR within the past 6 weeks
 */
function filterEligibleUsers(stats, collaboratorLogins) {
  const SIX_WEEKS_MS = 6 * 7 * 24 * 60 * 60 * 1000;
  const cutoff = new Date((new Date).getTime() - SIX_WEEKS_MS);
  const eligible = [];

  for (const [login, s] of stats.entries()) {
    if (s.open >= 5) continue;
    if (s.merged <= 2) continue;
    if (collaboratorLogins.has(login)) continue;
    if (s.open == 0 && s.mostRecentCreatedAt < cutoff) continue;
    if ((s.merged / s.closed) < 0.5) continue;

    eligible.push(login);
  }
  return eligible;
}

/**
 * Given a repo (owner/repo) and a GitHub token, returns the login names of
 * eligible external contributors matching all filters.
 *
 * @param {Object} params
 * @param {string} params.owner - Repo owner (user or org).
 * @param {string} params.repo - Repo name.
 * @param {string} params.token - GitHub token with repo read (and ideally push) access.
 * @returns {Promise<string[]>}
 */
async function findEligibleContributors({ owner, repo, token } = {}) {
  if (!owner || !repo) {
    throw new Error('findEligibleContributors requires both "owner" and "repo".');
  }
  if (!token) {
    throw new Error('findEligibleContributors requires a "token".');
  }

  const [pullRequests, collaboratorLogins] = await Promise.all([
    fetchPullRequests(owner, repo, token),
    fetchCollaborators(owner, repo, token),
  ]);

  const stats = aggregateUserStats(pullRequests);
  return filterEligibleUsers(stats, collaboratorLogins);
}

// ---- CLI entry point ----

function isRunAsCLI() {
  return import.meta.url === `file://${process.argv[1]}`;
}

async function runCLI() {
  const [, , owner, repo] = process.argv;
  if (!owner || !repo) {
    console.error('Usage: node find-eligible-contributors.js <owner> <repo>');
    process.exit(1);
  }
  const token = process.env.GITHUB_TOKEN;
  if (!token) {
    console.error('Error: set GITHUB_TOKEN environment variable with a valid GitHub token.');
    process.exit(1);
  }

  console.error(`Fetching pull requests and collaborators for ${owner}/${repo} in parallel...`);
  const eligibleLogins = await findEligibleContributors({ owner, repo, token });

  console.log(JSON.stringify(eligibleLogins, null, 2));
  console.error(`\n${eligibleLogins.length} user(s) match the criteria.`);
}

if (isRunAsCLI()) {
  runCLI().catch((err) => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}

export { findEligibleContributors };

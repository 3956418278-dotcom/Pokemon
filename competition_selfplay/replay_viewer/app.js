"use strict";

const state = { frames: [], index: 0, timer: null, names: new Map(), fileName: "" };
const $ = (id) => document.getElementById(id);
const energyNames = { 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M" };

function walkCards(value, visit) {
  if (!value || typeof value !== "object") return;
  if (!Array.isArray(value) && Number.isInteger(value.id)) visit(value);
  if (Array.isArray(value)) value.forEach((item) => walkCards(item, visit));
  else Object.values(value).forEach((item) => walkCards(item, visit));
}

function extractFrames(replay) {
  state.names.clear();
  const frames = [];
  (replay.steps || []).forEach((step, stepIndex) => {
    (step || []).forEach((agentStep, agentIndex) => {
      (agentStep?.visualize || []).forEach((visual, visualIndex) => {
        if (!visual?.current || !Array.isArray(visual.current.players)) return;
        walkCards(visual, (card) => {
          if (card.name) state.names.set(card.id, card.name);
        });
        frames.push({ ...visual, stepIndex, agentIndex, visualIndex });
      });
    });
  });
  if (!frames.length) throw new Error("JSON 中没有 visualize.current 动画帧");
  return frames;
}

function cardName(card) {
  if (!card) return "空";
  return card.name || state.names.get(card.id) || `Card ${card.id}`;
}

function renderCard(card) {
  const node = $("cardTemplate").content.firstElementChild.cloneNode(true);
  if (!card) {
    node.classList.add("empty");
    node.querySelector("strong").textContent = "空";
    return node;
  }
  node.querySelector("strong").textContent = cardName(card);
  node.querySelector(".serial").textContent = card.serial == null ? `#${card.id}` : `#${card.serial}`;
  const hp = card.hp;
  const maxHp = card.maxHp;
  const hpNode = node.querySelector(".hp");
  if (Number.isFinite(hp)) {
    hpNode.textContent = `HP ${hp}${Number.isFinite(maxHp) ? ` / ${maxHp}` : ""}`;
    if (Number.isFinite(maxHp) && maxHp > 0) {
      const bar = document.createElement("div");
      bar.className = "hp-bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.max(0, Math.min(100, hp / maxHp * 100))}%`;
      bar.append(fill);
      hpNode.append(bar);
    }
  }
  const energies = card.energies || (card.energyCards || []).map((item) => item.id);
  const energyList = node.querySelector(".energy-list");
  energies.forEach((energyId) => {
    const chip = document.createElement("span");
    const label = energyNames[energyId] || String(energyId);
    chip.className = `energy ${label.toLowerCase()}`;
    chip.textContent = label;
    energyList.append(chip);
  });
  return node;
}

function countZone(player, visibleName, countName) {
  if (Array.isArray(player?.[visibleName])) return player[visibleName].length;
  return player?.[countName] ?? "?";
}

function renderPlayer(player, index, current) {
  const root = $(`player${index}`);
  const active = player?.active?.[0] || null;
  const bench = player?.bench || [];
  root.innerHTML = "";
  const title = document.createElement("div");
  title.className = "player-title";
  title.innerHTML = `<h2>玩家 ${index}${current.yourIndex === index ? " · 当前视角" : ""}</h2>`;
  const stats = document.createElement("div");
  stats.className = "stats";
  const values = [
    ["牌库", player?.deckCount ?? countZone(player, "deck", "deckCount")],
    ["手牌", countZone(player, "hand", "handCount")],
    ["奖励", countZone(player, "prize", "prizeCount")],
    ["弃牌", countZone(player, "discard", "discardCount")],
    ["Bench", `${bench.length}/${player?.benchMax ?? 5}`],
  ];
  values.forEach(([name, value]) => {
    const chip = document.createElement("span");
    chip.textContent = `${name} ${value}`;
    stats.append(chip);
  });
  title.append(stats);
  root.append(title);

  const zones = document.createElement("div");
  zones.className = "zones";
  const activeZone = document.createElement("div");
  activeZone.innerHTML = '<div class="zone-label">ACTIVE</div>';
  activeZone.append(renderCard(active));
  const benchZone = document.createElement("div");
  benchZone.innerHTML = '<div class="zone-label">BENCH</div>';
  const list = document.createElement("div");
  list.className = "bench-list";
  bench.forEach((card) => list.append(renderCard(card)));
  benchZone.append(list);
  zones.append(activeZone, benchZone);
  root.append(zones);
}

function describeSelection(frame) {
  const select = frame.select || {};
  const selected = frame.selected || [];
  const lines = [
    `step=${frame.stepIndex} agent=${frame.agentIndex} visualize=${frame.visualIndex}`,
    `type=${select.type ?? "—"} context=${select.context ?? "—"}`,
    `effect=${cardName(select.effect)} contextCard=${cardName(select.contextCard)}`,
    `min=${select.minCount ?? "—"} max=${select.maxCount ?? "—"}`,
    `selected=[${selected.join(", ")}]`,
  ];
  if (frame.action != null) lines.push(`action=${JSON.stringify(frame.action)}`);
  return lines.join("\n");
}

function describeLog(log) {
  if (!log || typeof log !== "object") return String(log);
  const bits = [];
  if (log.playerIndex != null) bits.push(`P${log.playerIndex}`);
  if (log.cardId != null) bits.push(cardName({ id: log.cardId }));
  if (log.attackId != null) bits.push(`attack ${log.attackId}`);
  if (log.cardIdTarget != null) bits.push(`→ ${cardName({ id: log.cardIdTarget })}`);
  if (log.value != null) bits.push(`value ${log.value}`);
  bits.push(`type ${log.type ?? "?"}`);
  return bits.join(" · ") + `\n${JSON.stringify(log)}`;
}

function render() {
  if (!state.frames.length) return;
  const frame = state.frames[state.index];
  const current = frame.current;
  renderPlayer(current.players[0] || {}, 0, current);
  renderPlayer(current.players[1] || {}, 1, current);
  const stadium = current.stadium?.[0];
  $("stadium").textContent = stadium ? cardName(stadium) : "无场地";
  $("fileName").textContent = state.fileName;
  $("turnLabel").textContent = `${current.turn ?? "—"} · 行动 ${current.turnActionCount ?? "—"}`;
  $("frameLabel").textContent = `${state.index + 1} / ${state.frames.length}`;
  $("resultLabel").textContent = current.result === -1 ? "进行中" : `玩家 ${current.result} 获胜`;
  $("timeline").value = state.index;
  $("selection").textContent = describeSelection(frame);
  const logs = frame.logs || current.logs || [];
  $("logs").innerHTML = "";
  if (!logs.length) $("logs").textContent = "本帧无日志。";
  logs.forEach((log) => {
    const row = document.createElement("div");
    row.className = "log-row";
    row.textContent = describeLog(log);
    $("logs").append(row);
  });
}

function stop() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  $("playButton").textContent = "播放";
}

function play() {
  if (!state.frames.length) return;
  if (state.timer) return stop();
  if (state.index >= state.frames.length - 1) state.index = 0;
  $("playButton").textContent = "暂停";
  state.timer = window.setInterval(() => {
    if (state.index >= state.frames.length - 1) return stop();
    state.index += 1;
    render();
  }, Number($("speedSelect").value));
}

function move(delta) {
  if (!state.frames.length) return;
  stop();
  state.index = Math.max(0, Math.min(state.frames.length - 1, state.index + delta));
  render();
}

async function loadReplay(text, name) {
  stop();
  const replay = JSON.parse(text);
  state.frames = extractFrames(replay);
  state.index = 0;
  state.fileName = name;
  $("timeline").max = state.frames.length - 1;
  $("emptyState").hidden = true;
  render();
}

async function loadFile(file) {
  if (!file) return;
  try { await loadReplay(await file.text(), file.name); }
  catch (error) { window.alert(`无法加载 replay：${error.message}`); }
}

$("fileInput").addEventListener("change", (event) => loadFile(event.target.files[0]));
$("playButton").addEventListener("click", play);
$("prevButton").addEventListener("click", () => move(-1));
$("nextButton").addEventListener("click", () => move(1));
$("firstButton").addEventListener("click", () => move(-Infinity));
$("lastButton").addEventListener("click", () => move(Infinity));
$("timeline").addEventListener("input", (event) => { stop(); state.index = Number(event.target.value); render(); });
$("speedSelect").addEventListener("change", () => { if (state.timer) { stop(); play(); } });

const dropZone = $("dropZone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));
window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select")) return;
  if (event.code === "Space") { event.preventDefault(); play(); }
  if (event.code === "ArrowLeft") move(event.shiftKey ? -10 : -1);
  if (event.code === "ArrowRight") move(event.shiftKey ? 10 : 1);
});

const replayUrl = new URLSearchParams(window.location.search).get("replay");
if (replayUrl) {
  fetch(replayUrl)
    .then((response) => { if (!response.ok) throw new Error(`${response.status} ${response.statusText}`); return response.text(); })
    .then((text) => loadReplay(text, replayUrl.split("/").pop()))
    .catch((error) => window.alert(`URL replay 加载失败：${error.message}`));
}

const $ = (id) => document.getElementById(id);
let conversationId = 'default';
let busy = false;
const views = {chat: $('chatView'), docs: $('docsView'), memory: $('memoryView')};

function setTab(tab) {
  Object.entries(views).forEach(([key, el]) => el.classList.toggle('hidden', key !== tab));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  $('pageTitle').textContent = ({chat:'College AI Assistant', docs:'Knowledge base', memory:'Saved memory'})[tab];
  if (tab === 'docs') loadDocuments();
  if (tab === 'memory') loadMemories();
}
document.querySelectorAll('.nav-item').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab)));
$('newChat').addEventListener('click', async () => {
  conversationId = 'chat-' + Date.now();
  $('messages').innerHTML = '';
  $('welcome').classList.remove('hidden');
  setTab('chat');
});
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function addMessage(role, content, sources=[], mode='') {
  $('welcome').classList.add('hidden');
  const wrap = document.createElement('div'); wrap.className = `message ${role}`;
  const avatar = document.createElement('div'); avatar.className='message-avatar'; avatar.textContent = role === 'user' ? 'Y' : '✦';
  const body = document.createElement('div'); body.className='message-body';
  const meta = document.createElement('div'); meta.className='message-meta'; meta.textContent = role === 'user' ? 'YOU' : 'CAMPUSAI';
  const bubble = document.createElement('div'); bubble.className='bubble'; bubble.textContent=content;
  body.append(meta,bubble); wrap.append(avatar,body); $('messages').append(wrap);
  if (sources && sources.length) {
    const sourceWrap=document.createElement('div');sourceWrap.className='sources';
    const title=document.createElement('div');title.className='sources-title';title.textContent=`Sources retrieved (${sources.length})`;sourceWrap.append(title);
    sources.forEach(s=>{const card=document.createElement('div');card.className='source-card';card.innerHTML=`<b>${escapeHtml(s.document)}</b><small>Chunk ${escapeHtml(s.chunk)} · relevance ${escapeHtml(s.score)}</small><p>${escapeHtml(s.excerpt || '')}</p>`;sourceWrap.append(card)});
    $('messages').append(sourceWrap);
  }
  if (mode) { const m=document.createElement('div');m.className='mode-label';m.textContent=mode;$('messages').append(m); }
  $('messages').scrollTop=$('messages').scrollHeight;
}
async function sendMessage(text) {
  if (busy || !text.trim()) return;
  busy=true;$('sendButton').disabled=true;addMessage('user',text.trim());
  const wait=document.createElement('div');wait.className='message';wait.id='typing';wait.innerHTML='<div class="message-avatar">✦</div><div class="bubble">Searching documents and preparing an answer…</div>';$('messages').append(wait);
  try {
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text.trim(),conversation_id:conversationId})});
    const data=await r.json();if(!r.ok)throw new Error(data.detail||'Request failed');
    $('typing')?.remove();addMessage('assistant',data.content,data.sources||[],data.mode||'');
    if(data.warning){const w=document.createElement('div');w.className='mode-label';w.textContent=data.warning;$('messages').append(w)}
  } catch(e) {$('typing')?.remove();addMessage('assistant','Could not contact the backend. Make sure the server is running, then try again. Details: '+e.message)}
  finally {busy=false;$('sendButton').disabled=false;$('messageInput').focus()}
}
$('chatForm').addEventListener('submit',e=>{e.preventDefault();const v=$('messageInput').value;$('messageInput').value='';$('messageInput').style.height='auto';sendMessage(v)});
$('messageInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('chatForm').requestSubmit()}});
$('messageInput').addEventListener('input',e=>{e.target.style.height='auto';e.target.style.height=Math.min(e.target.scrollHeight,120)+'px'});
document.querySelectorAll('.suggestion').forEach(b=>b.addEventListener('click',()=>sendMessage(b.dataset.prompt)));
async function loadDocuments(){try{const r=await fetch('/api/documents');const d=await r.json();$('docCount').textContent=`${d.chunks} chunks`;$('documentList').innerHTML='';if(!d.documents.length){$('documentList').innerHTML='<p class="muted">No documents indexed yet.</p>';return}d.documents.forEach(name=>{const row=document.createElement('div');row.className='document-row';row.innerHTML=`<div class="file-icon">▤</div><div><b>${escapeHtml(name)}</b><small>Indexed and available for retrieval</small></div>`;$('documentList').append(row)})}catch(e){$('documentList').textContent='Could not load documents.'}}
async function uploadFile(file){if(!file)return;$('uploadStatus').textContent='Uploading and indexing…';const form=new FormData();form.append('file',file);try{const r=await fetch('/api/upload',{method:'POST',body:form});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Upload failed');$('uploadStatus').textContent=`Indexed ${d.filename}. Total chunks: ${d.indexed_chunks}`;loadDocuments()}catch(e){$('uploadStatus').textContent='Error: '+e.message}}
$('fileInput').addEventListener('change',e=>uploadFile(e.target.files[0]));
$('dropZone').addEventListener('dragover',e=>{e.preventDefault();$('dropZone').style.borderColor='#6558e8'});
$('dropZone').addEventListener('dragleave',()=>{$('dropZone').style.borderColor=''});
$('dropZone').addEventListener('drop',e=>{e.preventDefault();$('dropZone').style.borderColor='';uploadFile(e.dataTransfer.files[0])});
async function loadMemories(){try{const r=await fetch('/api/memory');const d=await r.json();$('memoryList').innerHTML='';if(!d.memories.length){$('memoryList').innerHTML='<p class="muted">No saved memories yet. Add a preference or useful context above.</p>';return}d.memories.forEach(m=>{const row=document.createElement('div');row.className='memory-row';row.innerHTML=`<span>${escapeHtml(m.text)}</span><button data-id="${escapeHtml(m.id)}">Delete</button>`;row.querySelector('button').addEventListener('click',()=>deleteMemory(m.id));$('memoryList').append(row)})}catch(e){$('memoryList').textContent='Could not load memories.'}}
$('memoryForm').addEventListener('submit',async e=>{e.preventDefault();const memory=$('memoryInput').value.trim();if(!memory)return;try{const r=await fetch('/api/memory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({memory})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Save failed');$('memoryInput').value='';loadMemories()}catch(e){alert(e.message)}});
async function deleteMemory(id){await fetch('/api/memory/'+encodeURIComponent(id),{method:'DELETE'});loadMemories()}
async function checkHealth(){try{const r=await fetch('/api/health');const d=await r.json();$('backendStatus').textContent=`Online · ${d.indexed_chunks} chunks`;$('backendStatus').previousElementSibling?.classList.add('online')}catch(e){$('backendStatus').textContent='Backend unavailable'}}
async function loadHistory(){try{const r=await fetch('/api/history?conversation_id='+encodeURIComponent(conversationId));const d=await r.json();if(d.messages?.length){$('welcome').classList.add('hidden');d.messages.forEach(m=>{if(m.role==='user')addMessage('user',m.content);else addMessage('assistant',m.content,m.sources||[],m.mode||'')})}}catch(e){}}
checkHealth();loadDocuments();

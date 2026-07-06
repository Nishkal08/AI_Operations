const { Router } = require('express');
const prisma         = require('../config/db');
const agentService   = require('../services/agentService');
const twilioService  = require('../services/twilioService');
const sseService     = require('../services/sseService');

const router = Router();

/**
 * POST /webhook/whatsapp/incoming — Twilio fires this on every incoming WhatsApp message
 */
router.post('/incoming', async (req, res) => {
  // Immediately respond 200 with empty TwiML — Twilio needs this fast and won't repeat it as message
  res.status(200).type('text/xml').send('<Response/>');

  try {
    const { From, Body, MessageSid } = req.body;
    if (!From || !Body) return;

    const phone = From.replace('whatsapp:', '');

    // Find lead by phone
    const lead = await prisma.lead.findFirst({ where: { phone } });
    if (!lead) {
      console.log(`[Webhook] Unknown sender: ${phone}`);
      return;
    }

    // Cancel pending follow-ups — lead replied
    await prisma.followUp.updateMany({
      where: { leadId: lead.id, status: 'pending' },
      data:  { status: 'cancelled' },
    });

    const conversation = await prisma.conversation.findUnique({
      where: { leadId: lead.id },
    });
    if (!conversation) {
      console.log(`[Webhook] No conversation for lead ${lead.id}`);
      return;
    }

    // Save incoming message
    await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role:           'lead',
        content:        Body,
        waMessageId:    MessageSid,
      },
    });

    await prisma.conversation.update({
      where: { id: conversation.id },
      data:  { lastMessageAt: new Date() },
    });

    // Notify frontend of the lead's new incoming message (in both AI and Human mode)
    sseService.push(lead.businessId, 'new_message', {
      leadId:         lead.id,
      conversationId: conversation.id,
      message:        Body,
      role:           'lead',
      leadName:       lead.name,
    });

    // If human is active, just return — message is already pushed
    if (conversation.isHumanActive) {
      return;
    }

    // Call AI agent
    const campaign = await prisma.campaign.findUnique({
      where:   { id: lead.campaignId },
      include: { kb: true },
    });

    // Resolve KB ID — fall back to business's first KB if campaign has none linked
    let resolvedKbId = campaign?.kbId;
    if (!resolvedKbId) {
      const fallbackKb = await prisma.knowledgeBase.findFirst({
        where: { businessId: lead.businessId },
        orderBy: { createdAt: 'asc' },
      });
      if (fallbackKb) {
        resolvedKbId = fallbackKb.id;
        console.log(`[Webhook] Campaign kbId was null — using fallback KB: ${resolvedKbId}`);
      }
    }

    const result = await agentService.callAgent({
      threadId:       phone,
      businessId:     lead.businessId,
      leadId:         lead.id,
      leadName:       lead.name,
      message:        Body,
      kbId:           resolvedKbId,
      campaignConfig: campaign,
    });

    // Save agent reply
    const agentMessage = await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role:           'agent',
        content:        result.reply,
      },
    });

    // Update lead qualification and conversation stage in DB
    await prisma.lead.update({
      where: { id: lead.id },
      data: {
        qualificationScore: result.qualification_score !== undefined && result.qualification_score !== null ? result.qualification_score : lead.qualificationScore,
        status:             result.needs_human ? 'hot' : (result.stage || 'pending'),
        intentSignals:      result.intent_signals ? JSON.stringify(result.intent_signals) : lead.intentSignals,
      },
    });

    await prisma.conversation.update({
      where: { id: conversation.id },
      data: {
        stage: result.needs_human ? 'handoff' : (result.stage || 'opener'),
      },
    });

    // Send reply via WhatsApp
    const twilioResult = await twilioService.sendMessage(phone, result.reply);
    if (twilioResult && twilioResult.sid) {
      await prisma.message.update({
        where: { id: agentMessage.id },
        data: {
          waMessageId: twilioResult.sid,
          waStatus: twilioResult.status || 'sent',
        },
      });
    }

    // Send images if any (max 2, with small delay)
    if (result.images_to_send && result.images_to_send.length > 0) {
      const images = result.images_to_send.slice(0, 2);
      for (let i = 0; i < images.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 800));
        let url = images[i];
        let caption = i > 0 ? `📸 Project Photo ${i+1}` : `📸 Project Photo`;
        
        if (images[i] && typeof images[i] === 'object') {
          url = images[i].url;
          caption = images[i].description || images[i].caption || caption;
        } else if (images[i] && typeof images[i] === 'string' && images[i].startsWith('{')) {
          try {
            const parsed = JSON.parse(images[i]);
            url = parsed.url;
            caption = parsed.description || parsed.caption || caption;
          } catch(e) {}
        }
        await twilioService.sendMessage(phone, caption, url);
      }
    }

    // Send brochure PDF if any
    if (result.brochure_url) {
      await new Promise(resolve => setTimeout(resolve, 500));
      const brochureMsg = `📋 Here's the complete project brochure with floor plans, pricing & specifications:\n${result.brochure_url}`;
      await twilioService.sendMessage(phone, brochureMsg);
    }

    // Notify via SSE
    sseService.push(lead.businessId, 'agent_replied', {
      leadId:             lead.id,
      conversationId:     conversation.id,
      reply:              result.reply,
      stage:              result.needs_human ? 'handoff' : (result.stage || 'opener'),
      status:             result.needs_human ? 'hot' : (result.stage || 'pending'),
      qualificationScore: result.qualification_score || 0,
      intentSignals:      result.intent_signals || [],
      leadName:           lead.name,
    });

    // Hot lead? Alert sales team
    if (result.needs_human) {
      sseService.push(lead.businessId, 'hot_lead', {
        leadId:   lead.id,
        leadName: lead.name,
        score:    result.qualification_score,
      });
    }

    // ── Analytics upsert ─────────────────────────────────
    // Increment daily CampaignAnalytics so the reply rate is non-zero.
    try {
      const todayStart = new Date();
      todayStart.setHours(0, 0, 0, 0);

      const existing = await prisma.campaignAnalytics.findFirst({
        where: {
          campaignId: lead.campaignId,
          date: { gte: todayStart },
        },
      });

      const isQualified = (result.qualification_score || 0) >= 3;

      if (existing) {
        await prisma.campaignAnalytics.update({
          where: { id: existing.id },
          data: {
            messagesSent:    { increment: 1 },
            repliesReceived: { increment: 1 },
            leadsQualified:  isQualified ? { increment: 1 } : undefined,
            ...(result.needs_human ? { handoffsTriggered: { increment: 1 } } : {}),
          },
        });
      } else {
        await prisma.campaignAnalytics.create({
          data: {
            campaignId:      lead.campaignId,
            date:            new Date(),
            messagesSent:    1,
            repliesReceived: 1,
            leadsQualified:  isQualified ? 1 : 0,
            handoffsTriggered: result.needs_human ? 1 : 0,
          },
        });
      }
    } catch (analyticsErr) {
      // Non-critical — don't let analytics failure break the webhook
      console.error('[Webhook] Analytics upsert failed:', analyticsErr.message);
    }
  } catch (err) {
    console.error('[Webhook] Error processing incoming message:', err);
  }
});


/**
 * POST /webhook/whatsapp/status — Delivery receipts from Twilio
 */
router.post('/status', async (req, res) => {
  res.sendStatus(200);

  try {
    const { MessageSid, MessageStatus, ErrorCode, ErrorMessage } = req.body;
    if (!MessageSid || !MessageStatus) return;

    console.log(`[Twilio Status] SID: ${MessageSid} | Status: ${MessageStatus} | ErrorCode: ${ErrorCode || 'None'} | ErrorMessage: ${ErrorMessage || 'None'}`);

    await prisma.message.updateMany({
      where: { waMessageId: MessageSid },
      data:  { waStatus: MessageStatus },
    });
  } catch (err) {
    console.error('[Webhook] Error processing status update:', err);
  }
});

module.exports = router;

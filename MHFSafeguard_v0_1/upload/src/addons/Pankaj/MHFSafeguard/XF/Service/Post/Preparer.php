<?php

namespace Pankaj\MHFSafeguard\XF\Service\Post;

use Pankaj\MHFSafeguard\Content\ContentContext;
use Pankaj\MHFSafeguard\Pipeline\ModerationPipeline;

class Preparer extends XFCP_Preparer
{
    protected $mhfsScanPacket = null;
    protected $mhfsContext = null;

    public function runMhfSafeguardScan(): void
    {
        $options = \XF::options();

        if (empty($options->mhfsEnabled))
        {
            return;
        }

        $post = $this->post;
        $thread = $post->Thread;

        if (!$thread)
        {
            return;
        }

        $excludedForums = $options->mhfsExcludedForums ?: [];
        if (!is_array($excludedForums))
        {
            $excludedForums = [];
        }

        if (!empty($thread->node_id) && in_array((int)$thread->node_id, array_map('intval', $excludedForums), true))
        {
            return;
        }

        $isFirstPost = method_exists($post, 'isFirstPost') ? (bool)$post->isFirstPost() : false;
        $message = $isFirstPost ? ((string)$thread->title . "\n" . (string)$post->message) : (string)$post->message;

        $context = new ContentContext([
            'contentType' => 'post',
            'contentId' => (int)($post->post_id ?? 0),
            'threadId' => (int)($post->thread_id ?? $thread->thread_id ?? 0),
            'nodeId' => (int)($thread->node_id ?? 0),
            'userId' => (int)($post->user_id ?? \XF::visitor()->user_id),
            'username' => (string)($post->username ?? \XF::visitor()->username),
            'title' => (string)($thread->title ?? ''),
            'message' => $message,
            'isFirstPost' => $isFirstPost
        ]);

        $pipeline = new ModerationPipeline();
        $packet = $pipeline->scan($context);

        $this->mhfsContext = $context;
        $this->mhfsScanPacket = $packet;

        $finalAction = $packet['final_action'] ?? 'allow';

        if ($finalAction === 'moderate')
        {
            if ($isFirstPost)
            {
                $thread->discussion_state = 'moderated';
            }
            else
            {
                $post->message_state = 'moderated';
            }
        }
        else if ($finalAction === 'revise')
        {
            throw new \XF\PrintableException(\XF::phrase('mhfs_please_revise_message'));
        }
    }

    public function afterInsert()
    {
        parent::afterInsert();
        $this->recordMhfSafeguardScan();
    }

    public function afterUpdate()
    {
        parent::afterUpdate();
        $this->recordMhfSafeguardScan();
    }

    protected function recordMhfSafeguardScan(): void
    {
        if (!$this->mhfsContext || !$this->mhfsScanPacket)
        {
            return;
        }

        $post = $this->post;
        $thread = $post->Thread;

        $context = $this->mhfsContext
            ->withContentId((int)($post->post_id ?? 0))
            ->withThreadId((int)($post->thread_id ?? $thread->thread_id ?? 0));

        $pipeline = new ModerationPipeline();
        $pipeline->record(
            $context,
            (string)($this->mhfsScanPacket['clean_message'] ?? ''),
            (array)($this->mhfsScanPacket['result'] ?? []),
            (string)($this->mhfsScanPacket['final_action'] ?? 'allow')
        );
    }
}
